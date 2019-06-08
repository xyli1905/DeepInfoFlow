from base_options import BaseOption

class Distributed(object):
	def __init__(self, _num_of_worker, _max_epoch):
		self.num_of_worker = _num_of_worker
		self.max_epoch = _max_epoch
		self.opt = BaseOption().parse()
		self.opt.log_seperator[-1] = _max_epoch

	def get_epoch_distributed_range(self):
		
		log_seperator = self.opt.log_seperator
		log_frequency = self.opt.log_frequency
		workload_distributed_range = []
		for i, v in enumerate(log_seperator):
			if i == 0:
				workload_distributed_range.append(v / log_frequency[i])
			else:
				workload_distributed_range.append((v - log_seperator[i - 1]) / (log_frequency[i]))
		return workload_distributed_range


	def devide_workload(self):
		
		workload_distributed_range = self.get_epoch_distributed_range()
		total_workload = sum(workload_distributed_range)
		num_of_worker = self.num_of_worker 
		min_epoch, max_epoch = 1, self.max_epoch
		individual_workloads = [int(total_workload / num_of_worker) for i in range(num_of_worker - 1)]
		individual_workloads += [int(total_workload - sum(individual_workloads))]
		individual_work_ranges = []

		for i, workload in enumerate(individual_workloads):
			if i == 0: start, end = min_epoch, 0
			for j, workload_range in enumerate(workload_distributed_range):
				if workload_range == 0: continue
				if i == 0:
					if workload > workload_range:
						workload -= workload_range
						workload_distributed_range[j] = 0
						end += self.opt.log_seperator[j]
					else:
						end += workload * self.opt.log_frequency[j]
						break

				elif i == len(individual_workloads) - 1:
					start = individual_work_ranges[i - 1][1] + 1
					end = max_epoch
					break

				else:
					start = individual_work_ranges[i - 1][1] + 1
					end = individual_work_ranges[i - 1][1]
					if workload > workload_range:
						workload -= workload_range
						workload_distributed_range[j] = 0
						end += self.opt.log_seperator[j]
					else:
						end += workload * self.opt.log_frequency[j]
						break

			individual_work_ranges.append((start, end))

		return individual_work_ranges

	def conbine_results(self):
		pass


if __name__ == '__main__':
	d = Distributed(_num_of_worker=3, _max_epoch=4000)
	individual_work_ranges = d.devide_workload()
	print(individual_work_ranges)
