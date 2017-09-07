import numpy as np


def line_to_list_element(line):
	line = line.strip()
	splitted_line = line.split("\t")
	return splitted_line[0].strip(), splitted_line[-1].strip()


def normalize(array):
	array = np.copy(array)
	array[array < 0] = 0
	array = array / np.sum(array)
	return array


def get_score(array):
	new_array = np.copy(array)
	new_array[new_array < 0] = 0
	return np.sum(new_array)


def main(datapath, base_w, lost_w, win_w):
	with open(datapath, "r") as f:
		list_vocab = [line_to_list_element(line) for line in f]

	# Each word has points that determine the probability of being chosen.
	weights = np.array([base_w for _ in list_vocab])

	np.random.seed(None)
	integers = list(range(len(weights)))

	for _ in range(1000):
		i = np.random.choice(integers, p=normalize(weights))

		print("Your current score is", get_score(weights))

		usr_input = input("What's the japanese translation of" + list_vocab[i][0])
		if usr_input == list_vocab[i][1]:
			# Win
			weights[i] -= win_w
		else:
			# Loose
			weights[i] += lost_w
		if get_score(weights) == 0:
			print("you won!")
			break

if __name__ == "__main__":
	main("./data.txt", 1.5, 2, 1)
