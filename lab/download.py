"""Download official MaleCNS inputs and verify the versions used by this lab."""
import hashlib
import urllib.request
from .core import DATA
BASE='https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/flat-connectome/'
FILES={
 'annotations':('body-annotations-male-cns-v1.0-minconf-0.5.feather','2177e246113e4cfbf1e7772ec37c6da1955ff22e8063d0b1f833101f99a9a3b2'),
 'neurotransmitters':('body-neurotransmitters-male-cns-v1.0.feather','95c9289220663abeb3409f3ad9e5a7f8a53f8093f5139d15502cd08da8879621'),
 'edges':('connectome-weights-male-cns-v1.0-minconf-0.5.feather','e35da783d1c686b2b58b3b87cd6a403ae43bfcfba8bff28e08ef752c1a56afc1'),
}
def main():
    DATA.mkdir(parents=True,exist_ok=True)
    for name,(filename,digest) in FILES.items():
        target=DATA/(name+'.feather')
        candidate=target
        if not target.exists():
            print('Downloading',filename,flush=True)
            candidate=target.with_suffix('.download')
            urllib.request.urlretrieve(BASE+filename,candidate)
        with candidate.open('rb') as f: actual=hashlib.file_digest(f,'sha256').hexdigest()
        if actual!=digest: raise RuntimeError(f'Checksum mismatch: {candidate}; original retained for inspection')
        if candidate!=target: candidate.replace(target)
        print('Verified',name,flush=True)
if __name__=='__main__': main()
