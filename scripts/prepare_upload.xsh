for f in pathlib.Path(".").glob("*/model"):
    model_name = f.parent.name.split("+")[0]
    $XZ_OPT="-9 -T0" tar \
        -C @(f.parent) \
        --transform f"s|model|{model_name}|" --show-transformed-names \
        -cvJf tars/@(model_name).tar.xz \
        model