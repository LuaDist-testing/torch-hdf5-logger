-- This file was automatically generated for the LuaDist project.

package = 'torch-hdf5-logger'
version = '0.1-1'

description = {
  summary = 'Torch Hdf5 Logger',
  detailed = [[
    A logger which outputs to hdf5 files
  ]]
}

-- LuaDist source
source = {
  tag = "0.1-1",
  url = "git://github.com/LuaDist-testing/torch-hdf5-logger.git"
}
-- Original source
-- source = {
--   url = 'git://github.com/gpleiss/torch-hdf5-logger',
--   tag = 'v0.1-1',
-- }

dependencies = {
  'torch >= 7.0',
  'lua ~> 5.1',
  'moses',
  'hdf5'
}
build = {
  type = 'builtin',
  modules = {
    Hdf5Logger = 'hdf5-logger.lua'
  }
}