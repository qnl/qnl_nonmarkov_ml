class DataFilters:

    def __init__(self):
        pass

    @staticmethod
    def correct_timestep(d):
        for i, key in enumerate(list(d.keys())):
            if key != 't_0':
                grab_n = int(d[key]['Tm']/d[key]['integration_time'])
                d[key]['I'] = d[key]['I'][:, :grab_n]
                d[key]['Q'] = d[key]['Q'][:, :grab_n]

        return d
