from tqdm import tqdm

f = open("data_links/data.txt", "r")
data_links = f.read().split('\n')
f = open("data_links/label.txt", "r")
label_links = f.read().split('\n')

f = open("data_links/data_full.txt", "r")
data_all_links = f.read().split('\n')
f = open("data_links/label_full.txt", "r")
label_all_links = f.read().split('\n')

exists = []
datas = []
labels = []

data_object = {}
for _audio_link, _label in zip(data_links + data_all_links, label_links + label_all_links):
    if len(_audio_link) > 0 and len(_label) > 0:
        [origin_link, positive_link, _] = _audio_link.split(" ")
        [negative_link, _] = _label.split(" ")
        total_link = origin_link.split(".")[0] + "_" + positive_link.split(".")[0] + "_" + negative_link.split(".")[0]
        if total_link not in data_object:
            data_object[total_link] = 1
        else:
            data_object[total_link] += 1

with tqdm(total=len(data_links + data_all_links)) as pbar:
    for _audio_link, _label in zip(data_links + data_all_links, label_links + label_all_links):
        if len(_audio_link) > 0 and len(_label) > 0:
            [origin_link, positive_link, _] = _audio_link.split(" ")
            [negative_link, _] = _label.split(" ")
            total_link = origin_link.split(".")[0] + "_" + positive_link.split(".")[0] + "_" + negative_link.split(".")[0]
            if data_object[total_link] == 1:
                datas.append(_audio_link)
                labels.append(_label)
        pbar.update(1)


print(len(datas))
print(len(labels))

with open('data_links/data_merge.txt', 'w') as f:
    for item in datas:
        f.write("%s\n" % item)

with open('data_links/label_merge.txt', 'w') as f:
    for item in labels:
        f.write("%s\n" % item)
