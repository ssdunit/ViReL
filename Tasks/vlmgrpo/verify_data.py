from data_util import load_robo2vlm
from config import SYSTEM_PROMPT
SYS = SYSTEM_PROMPT
tok = None                        

train = load_robo2vlm(
    dataset_name="keplerccc/Robo2VLM-1",
    system_prompt=SYS,
    tokenizer=tok,
    split="train",
)

print(f"rows: {len(train)}")

bad = sum(1 for r in train if r["images"][0] is None)
print(f"rows with missing image: {bad}")

# also check for genuinely broken images, not just None
broken = 0
for r in train:
    img = r["images"][0]
    if img is not None:
        try:
            img.load()
        except Exception:
            broken += 1
print(f"rows with unreadable image data: {broken}")
