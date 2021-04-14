import ray
import graphity.pipelines

if __name__ == "__main__":
	#ray.init(address='auto')
	glass_shape = (3, 3)
	beta = 2.9
	task_count = 2
	tasks = [graphity.pipelines.create_task(idx, beta, glass_shape) for idx in range(task_count)]
	eq_graphs = graphity.pipelines.sync_evolver(tasks).run()
	aug_graphs = graphity.pipelines.sync_augmenter(eq_graphs, beta, sweeps=1).run()
	print(len(aug_graphs))
	print(f"beta = {beta}")
