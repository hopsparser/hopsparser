mkdir tars
for f in pg`.*/model`:
    model_name = f.parent.name.split("+")[0]
    print(f"Compressing model {model_name}")
    tar \
        --create \
        --directory @(f.parent) \
        --file tars/@(model_name).tar.xz \
        --format pax \
        --transform f"s|model|{model_name}|" --show-transformed-names \
        --use-compress-program "zstd --ultra -22" \
        --verbose \
        "model"