import argparse
import re
import signal
import sqlite3
import subprocess
import sys
import time

import matplotlib.pyplot as plt

def get_children(pid):
	pids = subprocess.run(["ps", "h", "--ppid", "%i"%pid, "-o", "pid"], stdout=subprocess.PIPE).stdout.decode('utf-8')
	pids = [int(pid.strip()) for pid in pids.split("\n") if pid]
	return pids

def get_memory_usage(pid):
	rss = re.compile("VmRSS:\s+(\d+)?.*")
	out = subprocess.run(["cat", f"/proc/{pid}/status"], stdout=subprocess.PIPE).stdout.decode('utf-8')
	out = int(re.search(rss, out).group(1))
	return 1024 * out

def plot(args):
	nums = []
	con = sqlite3.connect(args.log)
	cur = con.cursor()
	for row in cur.execute('SELECT * FROM logs ORDER BY time'): 
		v = row[1]/(1024**3)
		print(v)
		nums.append(v)

	plt.plot(range(len(nums)), nums)
	plt.title('title name')
	plt.xlabel('Runtime (seconds)')
	plt.ylabel('Memory usage (GiB)')
	plt.show()

def snoop(args):
	pids = get_children(args.pid)
	print(pids)
	con = sqlite3.connect(args.log)
	cur = con.cursor()
	t = 0

	# Create table
	cur.execute('DROP TABLE IF EXISTS logs')
	cur.execute('CREATE TABLE logs (time integer, mem_size integer)')
	def noleak_file(sig, frame):
		con.close()
		sys.exit(0)
	signal.signal(signal.SIGINT, noleak_file)

	try:
		while True:
			t += 1
			mem_usage = 0
			for pid in pids:
				mem_usage += get_memory_usage(pid)
			print(mem_usage)
			cur.execute(f"INSERT INTO logs VALUES ({t}, {mem_usage})")
			con.commit()
			time.sleep(1)
	except Exception:
		con.close()


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description='Snoop on memory usage of a process and children.')
	subparsers = parser.add_subparsers(help='Pick a mode of operation: Plot or Snoop', dest="command")
	parse_snoop = subparsers.add_parser('snoop', help="Record the memory usage of a process.",)
	parse_snoop.add_argument('--pid', dest='pid', type=int, required=True, help='PID of parent process.')
	parse_snoop.add_argument('--log', dest='log', type=str, required=True, help='Location to dump mem usage data.')
	parse_plot = subparsers.add_parser('plot', help="Record the memory usage of a process.")
	parse_plot.add_argument('--log', dest='log', type=str, required=True, help='Location to dump mem usage data.')

	args = parser.parse_args()
	if args.command == "plot": plot(args)
	elif args.command == "snoop": snoop(args)