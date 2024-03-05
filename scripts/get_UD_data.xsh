mkdir -p "local"
curl --output "local/ud-treebanks-v2.13.tgz" "https://lindat.mff.cuni.cz/repository/xmlui/bitstream/handle/11234/1-5287/ud-treebanks-v2.13.tgz"
tar -xvf "local/ud-treebanks-v2.13.tgz" -C "local"
rm "local/ud-treebanks-v2.13.tgz"
mkdir -p "local/UD_French-all"
for part in ("train", "dev", "test"):
    cat gf`local/ud-treebanks-v2.13/UD_French*/*{part}.conllu` > @(f"local/UD_French-all/fr_all-ud-{part}.conllu")