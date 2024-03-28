# Pytorch-MIL
```bash
git submodule sync
git submodule update --init
```

## To compile server code with kernels

```bash
make codebase
make install-server
```

You can debug/edit code in the build folder. When done, use python copy_back.py to copy changes back to the original src folder.


## To compile all

```bash
make install
```

## To test Punica code

```bash
HF_HUB_ENABLE_HF_TRANSFER=1 pytest -s -vv --disable-pytest-warnings -m "punica_test" build/server/tests
```