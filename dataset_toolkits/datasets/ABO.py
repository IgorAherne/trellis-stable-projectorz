import os
import re
import argparse
import tarfile
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import pandas as pd
from utils import get_file_hash


def add_args(parser: argparse.ArgumentParser):
    pass


def get_metadata(**kwargs):
    metadata = pd.read_csv("hf://datasets/jetx/TRELLIS-500K/ABO.csv")
    return metadata
        

def download(metadata, output_dir, **kwargs):    
    os.makedirs(os.path.join(output_dir, 'raw'), exist_ok=True)

    if not os.path.exists(os.path.join(output_dir, 'raw', 'abo-3dmodels.tar')):
        try:
            os.makedirs(os.path.join(output_dir, 'raw'), exist_ok=True)
            os.system(f"wget -O {output_dir}/raw/abo-3dmodels.tar https://amazon-berkeley-objects.s3.amazonaws.com/archives/abo-3dmodels.tar")
        except:
            print("\033[93m")
            print("Error downloading ABO dataset. Please check your internet connection and try again.")
            print("Or, you can manually download the abo-3dmodels.tar file and place it in the {output_dir}/raw directory")
            print("Visit https://amazon-berkeley-objects.s3.amazonaws.com/index.html for more information")
            print("\033[0m")
            raise FileNotFoundError("Error downloading ABO dataset")
    
    downloaded = {}
    metadata = metadata.set_index("file_identifier")
    with tarfile.open(os.path.join(output_dir, 'raw', 'abo-3dmodels.tar')) as tar:
        with ThreadPoolExecutor(max_workers=1) as executor, \
            tqdm(total=len(metadata), desc="Extracting") as pbar:
            def worker(instance: str) -> str:
                try:
                    tar.extract(f"3dmodels/original/{instance}", path=os.path.join(output_dir, 'raw'))
                    sha256 = get_file_hash(os.path.join(output_dir, 'raw/3dmodels/original', instance))
                    pbar.update()
                    return sha256
                except Exception as e:
                    pbar.update()
                    print(f"Error extracting for {instance}: {e}")
                    return None
                
            sha256s = executor.map(worker, metadata.index)
            executor.shutdown(wait=True)

    for k, sha256 in zip(metadata.index, sha256s):
        if sha256 is not None:
            if sha256 == metadata.loc[k, "sha256"]:
                downloaded[sha256] = os.path.join('raw/3dmodels/original', k)
            else:
                print(f"Error downloading {k}: sha256s do not match")

    return pd.DataFrame(downloaded.items(), columns=['sha256', 'local_path'])


def foreach_instance(metadata, output_dir, func, max_workers=None, desc='Processing objects') -> pd.DataFrame:
    import os
    from concurrent.futures import ThreadPoolExecutor
    from tqdm import tqdm
    
    # load metadata
    metadata = metadata.to_dict('records')

    # processing objects
    records = []
    max_workers = max_workers or os.cpu_count()
    try:
        with ThreadPoolExecutor(max_workers=max_workers) as executor, \
            tqdm(total=len(metadata), desc=desc) as pbar:
            def worker(metadatum):
                try:
                    local_path = metadatum['local_path']
                    sha256 = metadatum['sha256']
                    file = os.path.join(output_dir, local_path)
                    record = func(file, sha256)
                    if record is not None:
                        records.append(record)
                    pbar.update()
                except Exception as e:
                    print(f"Error processing object {sha256}: {e}")
                    pbar.update()
            
            executor.map(worker, metadata)
            executor.shutdown(wait=True)
    except:
        print("Error happened during processing.")
        
    return pd.DataFrame.from_records(records)
