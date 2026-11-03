"""Lead-reviewed extraction of the preserved KIT archive; no models or splits."""
from pathlib import Path
import zipfile, io, re, json, hashlib, datetime, math
from collections import Counter
import numpy as np
from lxml import etree
from scipy.spatial import cKDTree

HERE = Path(__file__).resolve().parent
ARCH = HERE.parents[1] / 'independent_evidence_20261030/kit_acquisition/Daten2018.zip'
COLS = {'A':'timestamp','C':'x-accel','D':'y-accel','E':'z-accel','F':'x-gyro','G':'y-gyro','H':'z-gyro',
        'S':'gps_timestamp','T':'lat','U':'lon','V':'speed','AA':'Material','AB':'Event'}
PAT = re.compile(rb'<c\b[^>]*\br="(A|C|D|E|F|G|H|S|T|U|V|AA|AB)([0-9]+)"[^>]*>(.*?)</c>', re.S)
VPAT = re.compile(rb'<v>([^<]*)</v>')
NS = '{http://schemas.openxmlformats.org/spreadsheetml/2006/main}'

def hashfile(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda:f.read(1048576),b''): h.update(block)
    return h.hexdigest()

def parse(blob):
    with zipfile.ZipFile(io.BytesIO(blob)) as z:
        ss=[''.join(x.itertext()) for x in etree.fromstring(z.read('xl/sharedStrings.xml'))]
        sheet=next(n for n in z.namelist() if n.startswith('xl/worksheets/sheet') and n.endswith('.xml'))
        xml=z.read(sheet)
    nr=max(int(v) for v in re.findall(rb'<row\b[^>]*\br="([0-9]+)"',xml))-1
    out={c:np.full(nr,np.nan,dtype=np.float64) for c in COLS}
    header={}; cache={}
    for m in PAT.finditer(xml):
        col=m[1].decode(); row=int(m[2]); v=VPAT.search(m[3])
        if v is None: continue
        value=v[1]
        if row==1:
            header[col]=ss[int(value)]
            continue
        if col=='S':
            si=int(value)
            if si not in cache:
                try: cache[si]=datetime.datetime.fromisoformat(ss[si].replace(',','.').replace('Z','+00:00')).timestamp()
                except ValueError: cache[si]=math.nan
            out[col][row-2]=cache[si]
        else:
            try: out[col][row-2]=float(value)
            except ValueError: pass
    assert header==COLS, (header,COLS)
    assert np.isfinite(out['A']).all()
    return out,header

def project(lat,lon):
    return np.column_stack([(lon-8.4)*111320*math.cos(math.radians(49.0)),(lat-49.0)*111320])

def main():
    arrays={}; files=[]; headers={}; xy_by_file={}
    with zipfile.ZipFile(ARCH) as z:
        for name in z.namelist():
            if '/IMU' not in name or not name.endswith('.xlsx'): continue
            d,h=parse(z.read(name)); key=f'f{len(files):02d}'; headers[name]=h
            t=d['A']; lat=d['T']; lon=d['U']; lab=d['AB']; gps=d['S']
            a=np.column_stack([d[c] for c in ('C','D','E','F','G','H')])
            valid=np.isfinite(lat)&np.isfinite(lon)&(lat>=48)&(lat<=50)&(lon>=7)&(lon<=10)
            xy=project(lat,lon); dt=np.diff(t); step=np.linalg.norm(np.diff(xy,axis=0),axis=1)
            changed=np.flatnonzero(valid[1:]&valid[:-1]&(step>0))+1
            update_speed=[]
            # GPS updates must be timed against previous distinct valid GPS fix, not the 220Hz IMU step.
            previous=None; plausible=valid.copy()
            for ix in np.flatnonzero(valid):
                if previous is None: previous=ix; continue
                distance=float(np.linalg.norm(xy[ix]-xy[previous]))
                if distance==0: continue
                elapsed=gps[ix]-gps[previous]
                if not np.isfinite(elapsed) or elapsed<=0: elapsed=t[ix]-t[previous]
                velocity=distance/elapsed if elapsed>0 else math.inf
                update_speed.append(velocity)
                if velocity>55: plausible[ix]=False
                previous=ix
            vals=lab[np.isfinite(lab)].astype(int); counts=Counter(vals.tolist())
            intervals=[]
            pos=np.flatnonzero(lab==3)
            if len(pos):
                cuts=np.flatnonzero((np.diff(pos)!=1)|(np.diff(t[pos])>.1))+1
                for ids in np.split(pos,cuts):
                    i,j=int(ids[0]),int(ids[-1]); intervals.append({'start_row':i,'end_row':j,'start':float(t[i]),'end':float(t[j]),'seconds':float(t[j]-t[i])})
            known=valid&np.isin(lab,[1,3]); coords=xy[known]
            xy_by_file[key]=np.unique(np.round(coords/5).astype(np.int64),axis=0)*5.0
            gd=gps[np.isfinite(gps)]
            dates=Counter(datetime.datetime.fromtimestamp(v,datetime.timezone.utc).date().isoformat() for v in np.unique(gd))
            files.append({'file_id':key,'path':name,'rows':len(t),'labels':{str(k):v for k,v in sorted(counts.items())},
                'numeric_start':float(t[0]),'numeric_end':float(t[-1]),'numeric_clock_decreases':int((dt<0).sum()),
                'gps_dates_unique_fix_counts':dict(dates),'gps_clock_median_offset_seconds':float(np.median(t[np.isfinite(gps)]-gd)) if len(gd) else None,
                'gps_fraction':float(valid.mean()),'gps_distinct_5m_known_cells':len(xy_by_file[key]),
                'gps_update_speed_max_mps':float(max(update_speed,default=0)),
                'gps_update_speed_p99_mps':float(np.quantile(update_speed,.99)) if update_speed else None,
                'known_both_classes':bool(1 in counts and 3 in counts),'pothole_annotation_intervals':intervals,
                'finite_axes_fraction':float(np.isfinite(a).all(axis=1).mean())})
            for k,v in {'timestamp':t,'gps_timestamp':gps,'lat':lat,'lon':lon,'speed':d['V'],'material':d['AA'],
                        'label':lab,'axes6':a}.items(): arrays[f'{key}_{k}']=v
            print(f'{key} {len(t)} labels={dict(counts)} GPS dates={dict(dates)}',flush=True)
    pairs=[]
    for i,a in enumerate(files):
        pa=xy_by_file[a['file_id']]
        for b in files[i+1:]:
            pb=xy_by_file[b['file_id']]
            if not len(pa) or not len(pb): continue
            da=cKDTree(pb).query(pa,k=1)[0]; db=cKDTree(pa).query(pb,k=1)[0]
            nearest=float(da.min()); na=int((da<=50).sum()); nb=int((db<=50).sum())
            if nearest<=100:
                pairs.append({'a':a['file_id'],'b':b['file_id'],'min_known_distance_m':nearest,
                              'fraction_a_known_cells_within_50m':na/len(pa),'fraction_b_known_cells_within_50m':nb/len(pb)})
    totals=Counter()
    for f in files:totals.update(f['labels'])
    report={'status':'raw_audit_no_models_or_split','archive_sha256':hashfile(ARCH),'script_sha256':hashfile(Path(__file__)),
            'files':files,'label_totals':dict(totals),'overlap_pairs':pairs,
            'independent_road_count':'not determined; numeric clock and GPS clock disagree; spatial overlap recorded explicitly'}
    np.savez_compressed(HERE/'raw_arrays_verified.npz',**arrays)
    (HERE/'headers_verified.json').write_text(json.dumps(headers,indent=2),encoding='utf8')
    (HERE/'group_audit_verified.json').write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf8')
    print('Audit complete',flush=True)

if __name__=='__main__':main()
