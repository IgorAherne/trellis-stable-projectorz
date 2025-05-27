import os
import re
import argparse
import zipfile
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import pandas as pd
from utils import get_file_hash


def add_args(parser: argparse.ArgumentParser):
    pass


def get_metadata(**kwargs):
    metadata = pd.read_csv("hf://datasets/jetx/TRELLIS-500K/3D-FUTURE.csv")
    return metadata
        

def download(metadata, output_dir, **kwargs):    
    os.makedirs(output_dir, exist_ok=True)

    if not os.path.exists(os.path.join(output_dir, 'raw', '3D-FUTURE-model.zip')):
        print("\033[93m")
        print("3D-FUTURE have to be downloaded manually")
        print(f"Please download the 3D-FUTURE-model.zip file and place it in the {output_dir}/raw directory")
        print("Visit https://tianchi.aliyun.com/specials/promotion/alibaba-3d-future for more information")
        print("\033[0m")
        raise FileNotFoundError("3D-FUTURE-model.zip not found")
    
    downloaded = {}
    metadata = metadata.set_index("file_identifier")
    with zipfile.ZipFile(os.path.join(output_dir, 'raw', '3D-FUTURE-model.zip')) as zip_ref:
        all_names = zip_ref.namelist()
        instances = [instance[:-1] for instance in all_names if re.match(r"^3D-FUTURE-model/[^/]+/$", instance)]
        instances = list(filter(lambda x: x in metadata.index, instances))
        
        with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor, \
            tqdm(total=len(instances), desc="Extracting") as pbar:
            def worker(instance: str) -> str:
                try:
                    instance_files = list(filter(lambda x: x.startswith(f"{instance}/") and not x.endswith("/"), all_names))
                    zip_ref.extractall(os.path.join(output_dir, 'raw'), members=instance_files)
                    sha256 = get_file_hash(os.path.join(output_dir, 'raw', f"{instance}/image.jpg"))
                    pbar.update()
                    return sha256
                except Exception as e:
                    pbar.update()
                    print(f"Error extracting for {instance}: {e}")
                    return None
                
            sha256s = executor.map(worker, instances)
            executor.shutdown(wait=True)

    for k, sha256 in zip(instances, sha256s):
        if sha256 is not None:
            if sha256 == metadata.loc[k, "sha256"]:
                downloaded[sha256] = os.path.join("raw", f"{k}/raw_model.obj")
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
