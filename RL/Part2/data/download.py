from datasets import load_dataset, Dataset
import os

os.makedirs("./Part2/data/sft_data", exist_ok=True)

ds_2round = load_dataset("Kwai-Keye/Thyme-SFT", split="2round", streaming=True).take(333)
ds_comp = load_dataset("Kwai-Keye/Thyme-SFT", split="computation", streaming=True).take(333)
ds_single = load_dataset("Kwai-Keye/Thyme-SFT", split="wo_thinking_thyme_single_round", streaming=True).take(334)

ds_2round = Dataset.from_list(list(ds_2round))
ds_comp = Dataset.from_list(list(ds_comp))
ds_single = Dataset.from_list(list(ds_single))

ds_2round.to_json("./Part2/data/sft_data/2round.json")
ds_comp.to_json("./Part2/data/sft_data/comp.json")
ds_single.to_json("./Part2/data/sft_data/single.json")

print("Download done: 2round=", len(ds_2round), "comp=", len(ds_comp), "single=", len(ds_single))
