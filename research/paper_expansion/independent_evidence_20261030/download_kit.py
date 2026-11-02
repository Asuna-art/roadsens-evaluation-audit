import pathlib,concurrent.futures,hashlib,json,time,requests
B=pathlib.Path(__file__).resolve().parent/'kit_acquisition';B.mkdir(exist_ok=True)
URL='https://zenodo.org/records/1461243/files/Daten2018.zip?download=1';N=546803578;CHUNK=8*1024*1024
def get(i):
 start=i*CHUNK;end=min(N,start+CHUNK)-1;p=B/f'part_{i:03}.bin'
 if p.exists() and p.stat().st_size==end-start+1:return i
 for attempt in range(4):
  try:
   r=requests.get(URL,headers={'Range':f'bytes={start}-{end}'},timeout=(20,90));r.raise_for_status()
   if r.status_code!=206 or r.headers.get('Content-Range')!=f'bytes {start}-{end}/{N}' or len(r.content)!=end-start+1:raise ValueError((r.status_code,r.headers.get('Content-Range'),len(r.content)))
   p.write_bytes(r.content);return i
  except Exception as e:
   if attempt==3:raise
   time.sleep(1+attempt)
if __name__=='__main__':
 with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
  for f in concurrent.futures.as_completed([pool.submit(get,i) for i in range((N+CHUNK-1)//CHUNK)]):print('chunk',f.result(),'verified range',flush=True)
 final=B/'Daten2018.zip';tmp=B/'Daten2018.zip.partial'
 if final.exists():raise FileExistsError(final)
 h=hashlib.md5();hs=hashlib.sha256()
 with tmp.open('wb') as out:
  for i in range((N+CHUNK-1)//CHUNK):
   raw=(B/f'part_{i:03}.bin').read_bytes();out.write(raw);h.update(raw);hs.update(raw)
 if tmp.stat().st_size!=N or h.hexdigest()!='e9a4eb2da043c1eb3dbf57134cf8796b':raise ValueError('archive checksum failure')
 tmp.rename(final);report={'status':'download_verified','source':URL,'bytes':N,'md5':h.hexdigest(),'sha256':hs.hexdigest(),'range_validation':'every response206 with exact Content-Range and byte count'};(B/'download_manifest.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)
