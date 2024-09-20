# prepare a new release
pyrelease version:
    @if [ -n "$(git status --porcelain || echo "dirty")" ]; then echo "repo is dirty!"; exit 1; fi
    sed -i 's/^version = ".*"$/version = "{{ version }}"/g' music-mesa-tables-py/Cargo.toml
    git add music-mesa-tables-py/Cargo.toml
    git commit -m "python release {{ version }}"
    git tag -m "Release {{ version }}" -a -e "v{{ version }}"
    @echo "check last commit and ammend as necessary, then git push --follow-tags"
