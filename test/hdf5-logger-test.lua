local _ = require 'moses'
require 'hdf5-logger'

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

local new_logger = Hdf5Logger(_.append({'learning_rate'}, keys), filename)
print(new_logger.stats)
new_logger:load()
print(new_logger.stats)
