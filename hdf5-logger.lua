require 'hdf5'
local _ = require 'moses'
local Hdf5Logger = torch.class('Hdf5Logger')

function Hdf5Logger:__init(keys, filename, opts)
  opts = opts or {}

  self.stats = {}
  self.filename = filename
  if opts.timestamp then
    self.filename = self.filename .. '-' .. os.date('%Y_%m_%d_%X')
  end

  for i = 1, #keys do
    self.stats[keys[i]] = {
      iters =  {},
      vals = {},
    }
  end
end

function Hdf5Logger:load()
  local f = hdf5.open(self.filename, 'r')

  local data = f:all()
  for key, key_stats in pairs(data) do
    if #(self.stats[key].iters) ~= 0 or #(self.stats[key].vals) ~= 0 then
      error('Can only load to empty file')
    end

    self.stats[key].iters = data[key].iters:totable()

    local vals_data = data[key].vals
    if vals_data:nDimension() == 1 then
      self.stats[key].vals = vals_data:totable()
    else
      for i = 1, vals_data:size(1) do
        table.insert(self.stats[key].vals, vals_data[i])
      end
    end
  end
  f:close()
end

function Hdf5Logger:log(key, iter, val)
  if not _.has(self.stats, key) then
    error('Unknown key ' .. key)
  end

  if not (type(val) == 'userdata' or type(val) == 'number') then
    error('Unknown type for val')
  end

  table.insert(self.stats[key].iters, iter)
  table.insert(self.stats[key].vals, val)
end

function Hdf5Logger:write()
  os.remove(self.filename)

  local f = hdf5.open(self.filename, 'w')
  for key, key_stats in pairs(self.stats) do
    local iters = torch.Tensor(key_stats.iters)
    local vals = torch.Tensor(_.map(key_stats.vals, function(i, val)
      if type(val) == 'userdata' then
        return val:totable()
      else
        return val
      end
    end))

    if iters:storage() then
      f:write(key .. '/iters', iters)
      f:write(key .. '/vals', vals)
    end
  end
  f:close()
end
