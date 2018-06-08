# torch-hdf5-logger

A logger which outputs to hdf5 files.
It can log numbers or torch tensors.

Each value it logs is independent. E.g. - you can log training error every
1000 iterations, and validation error every 10000 iterations.

## Installation
`luarocks install torch-hdf5-logger`

## Usage

```lua
require 'Hdf5Logger'

local keys = {'loss', 'valid_error', 'conf_matrix'}
local filename = os.getenv('TEST_FILENAME')

local logger = Hdf5Logger(keys, filename)

logger:log('loss', 10, 0.9)
logger:log('valid_error', 10, 0.4)
logger:log('conf_matrix', 10, torch.rand(3,3))

logger:log('loss', 20, 0.8)
logger:log('valid_error', 20, 0.5)
logger:log('conf_matrix', 20, torch.rand(3,3))

logger:log('loss', 30, 0.7)
logger:log('valid_error', 30, 0.6)

logger:log('loss', 40, 0.6)

logger:write()
test/hdf5-logg
```

### Specifics
```
Hdf5Logger(keys, filenames, opts)
```

Creates a new logger.
- `keys` is the list of attributes you wish to log.
- `filename` the output filename.
- `opts.timestamp` - whether or not a timestamp is added to the filename.

```
Hdf5Logger:log(key, iter, val)
```

Adds a new log.
- `key` - name of key to log. *Key needs to be in list of keys provided to constructor.*
- `iter` - "timestamp" of value (e.g. number of iterations, number of epochs, etc.)
- `val` - value to be logged. **Number or tensor only**.

```
Hdf5Logger:write()
```

Write the log to an hdf5 file. It will overwrite any existing pre-existing file.
