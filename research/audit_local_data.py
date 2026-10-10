"""Read-only local metadata audit. Outputs aggregates only; no patient identifiers."""
import ast
import collections
import csv
import hashlib
import io
import json
from pathlib import Path
import re
import zipfile

ROOT = Path(r'D:\dataset\mimic-iv-ecg-diagnostic-electrocardiogram')
OUT = Path(r'D:\CAD_ML\research\local_data_aggregates.json')
result = {}

def read_csv(path):
    with path.open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))

acsroot = Path(r'D:\新建文件夹')
train = read_csv(acsroot / 'CSV/train.csv')
test = read_csv(acsroot / 'CSV/test.csv')
ids = [{r['Patient_id'] for r in rows} for rows in (train, test)]
labels = [k for k in train[0] if k not in test[0]]
counts = {k: dict(collections.Counter(r[k] for r in train)) for k in labels}
rowfiles = list((acsroot/'row_data').iterdir())
medfiles = list((acsroot/'med_data').iterdir())
headers = [p for p in rowfiles if p.suffix == '.hea']
header_shapes = collections.Counter()
for p in headers:
    with p.open() as f:
        first = f.readline().split()
    header_shapes[' '.join(first[1:4])] += 1
result['acs'] = {
    'train_rows': len(train), 'test_rows': len(test),
    'train_unique_patients': len(ids[0]), 'test_unique_patients': len(ids[1]),
    'patient_overlap': len(ids[0]&ids[1]),
    'unique_records': {split:len({r['ecg_row_record'] for r in rows}) for split,rows in [('train',train),('test',test)]},
    'train_fields': list(train[0]), 'test_fields': list(test[0]),
    'labels': counts,
    'omi_or_cto': sum(r['OMI']=='1' or r['CTO']=='1' for r in train),
    'omi_and_cto': sum(r['OMI']=='1' and r['CTO']=='1' for r in train),
    'ami_stemi_nstemi_inconsistent': sum((r['AMI']=='1') != (r['STEMI']=='1' or r['NSTEMI']=='1') for r in train),
    'any_diagnostic_ami_ua': sum(r['AMI']=='1' or r['UA']=='1' for r in train),
    'all_core_zero': sum(all(r[k]=='0' for k in ['AMI','OMI','CTO','PCI','NSTEMI','STEMI','UA','Prior_PCI']) for r in train),
    'waveform_files':dict(collections.Counter(p.suffix for p in rowfiles)),
    'median_files':dict(collections.Counter(p.suffix for p in medfiles)),
    'all_header_signal_rate_samples':dict(header_shapes),
    'median_file_sizes':dict(collections.Counter(p.stat().st_size for p in medfiles)),
    'missing_cells_train': {k:sum(not r[k].strip() for r in train) for k in train[0]},
}
with zipfile.ZipFile(r'D:\29925314.zip') as z:
    with zipfile.ZipFile(io.BytesIO(z.read('CSV.zip'))) as inner:
        matches = {}
        for name in inner.namelist():
            if name.endswith('.csv'):
                p = acsroot/'CSV'/Path(name).name
                matches[Path(name).name] = p.exists() and hashlib.sha256(inner.read(name)).digest() == hashlib.sha256(p.read_bytes()).digest()
        result['acs']['archive_csv_equals_extracted'] = matches

mimic_name = 'mimic-iv-ecg-diagnostic-electrocardiogram-matched-subset-1.0'
mimic = ROOT/mimic_name/mimic_name
subjects = set()
studies = set()
total = 0
missing_paths = 0
with (mimic/'record_list.csv').open(encoding='utf-8-sig',newline='') as f:
    reader = csv.DictReader(f)
    fields = reader.fieldnames
    first_record = None
    for row in reader:
        if first_record is None: first_record = row
        total += 1
        subjects.add(row['subject_id'])
        studies.add(row['study_id'])
with (mimic/'machine_measurements.csv').open(encoding='utf-8-sig',newline='') as f:
    reader = csv.DictReader(f)
    machine_fields = reader.fieldnames
    machine_rows = sum(1 for _ in reader)
with (mimic/'waveform_note_links.csv').open(encoding='utf-8-sig',newline='') as f:
    reader = csv.DictReader(f)
    note_fields = reader.fieldnames
    note_rows = sum(1 for _ in reader)
sample_hea = (mimic/first_record['path']).with_suffix('.hea')
lines = sample_hea.read_text().splitlines()
result['mimic'] = {
    'metadata_root':str(mimic), 'record_fields':fields,'records':total,
    'unique_subjects':len(subjects),'unique_studies':len(studies),
    'machine_rows':machine_rows,'machine_fields':machine_fields,
    'note_link_rows':note_rows,'note_fields':note_fields,
    'sample_signal_rate_samples':lines[0].split()[1:4],
    'sample_leads':[x.split()[-1] for x in lines[1:13]],
    'top_level_files':[p.name for p in mimic.iterdir() if p.is_file()],
    'machine_data_dictionary': (mimic/'machine_measurements_data_dictionary.csv').read_text(),
}

ptb_name='ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3'
ptb=ROOT/ptb_name/ptb_name
ptbrows=read_csv(ptb/'ptbxl_database.csv')
scp=read_csv(ptb/'scp_statements.csv')
codecounts=collections.Counter(k for row in ptbrows for k in ast.literal_eval(row['scp_codes']))
result['ptbxl']={
    'metadata_root':str(ptb),'rows':len(ptbrows),'unique_patients':len({r['patient_id'] for r in ptbrows}),
    'unique_ecgs':len({r['ecg_id'] for r in ptbrows}),
    'fields':list(ptbrows[0]),'fold_counts':dict(collections.Counter(r['strat_fold'] for r in ptbrows)),
    'diagnostic_classes':dict(collections.Counter(row['diagnostic_class'] for row in scp if row['diagnostic']=='1.0')),
    'code_counts':dict(codecounts),'scp_fields':list(scp[0]),
    'header_shapes':{key: (ptb/(ptbrows[0][key]+'.hea')).read_text().splitlines()[0].split()[1:4] for key in ['filename_lr','filename_hr']},
    'top_level_files':[p.name for p in ptb.iterdir() if p.is_file()],
}

gu=Path(r'D:\新建文件夹 (2)\Pre-inflation, Inflation, Post-inflation ECG Recordings')
result['gu']={}
for p in gu.glob('*.zip'):
    with zipfile.ZipFile(p) as z:
        files=[i for i in z.infolist() if not i.is_dir()]
        result['gu'][p.name]={
            'files':len(files),'extensions':dict(collections.Counter(Path(i.filename).suffix for i in files)),
            'uncompressed_bytes':sum(i.file_size for i in files),
            'path_depth_counts':dict(collections.Counter(len(Path(i.filename).parts) for i in files)),
            'filename_patterns':dict(collections.Counter(re.sub('[0-9]','#',Path(i.filename).name) for i in files)),
        }
OUT.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(result,ensure_ascii=False,indent=2))
