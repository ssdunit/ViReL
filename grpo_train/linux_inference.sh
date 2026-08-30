skip_domains=()

domains=$(python -c "
from datasets import load_dataset
ds = load_dataset('sy-xie/robovista')
split = list(ds.keys())[0]
for d in sorted(set(ds[split]['domain'])):
    print(d)
")

while IFS= read -r domain; do
    [ -z "$domain" ] && continue
    skip=false
    for s in "${skip_domains[@]}"; do
        [ "$domain" = "$s" ] && skip=true && break
    done
    [ "$skip" = true ] && continue

    echo "Running domain: $domain"
    python evaluation.py --mode robovista --greedy --repetition_penalty 0.5 --domain "$domain"
done <<< "$domains"
