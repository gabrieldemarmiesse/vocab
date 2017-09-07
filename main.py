import numpy as np
import os
import math

tab_value = 4


def cls():
	os.system('cls' if os.name == 'nt' else 'clear')


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


def closeness(string_1, string_2):
	total = 0.
	for letter1, letter2 in zip(string_1, string_2):
		if letter1 == letter2:
			total += 1
	return total / len(string_1)


def print_clear(usr_input, truth):
	if len(usr_input) == len(truth):
		if closeness(usr_input, truth) > 0.6:
			to_display = [letter for letter in truth]
			for i in range(len(truth)):
				if truth[i] != usr_input[i]:
					to_display[i] = to_display[i].upper()
			return "".join(to_display)
	return truth


def print_scores(vocab_list, weights):
	for voc, weight in zip(vocab_list, weights):
		nb_tabs_ = (28. - len(voc[0])) / float(tab_value)
		nb_tabs = math.ceil(nb_tabs_)
		print(voc[0] + "\t" * nb_tabs + str(weight))


def main(datapath, base_w, lost_w, win_w):
	with open(datapath, "r") as f:
		list_vocab = [line_to_list_element(line) for line in f]

	# Each word has points that determine the probability of being chosen.
	weights = np.array([base_w for _ in list_vocab])

	np.random.seed(None)
	integers = list(range(len(weights)))

	for _ in range(1000):
		i = np.random.choice(integers, p=normalize(weights))

		print_scores(list_vocab, weights)
		print("\nYour current score is", get_score(weights), "\n")

		usr_input = input("What's the japanese translation of \"" + list_vocab[i][0] + "\"" + "\n")
		usr_input = usr_input.strip()

		if usr_input == list_vocab[i][1]:
			# Win
			weights[i] -= win_w
			print("Good answer!")
			if weights[i] <= 0:
				print("you've finished the word", list_vocab[i])
				weights[i] = 0
			input()
		else:
			# Loose
			weights[i] += lost_w
			print("The right answer was:", print_clear(usr_input, list_vocab[i][1]))
			input()
		cls()
		if get_score(weights) == 0:
			print("you won!")
			break


if __name__ == "__main__":
	main("./data.txt", 1.5, 1.5, 1)
