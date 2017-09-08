import numpy as np
import os
import math
from glob import glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

tab_value = 4
fullprint = False


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
	if fullprint:
		for voc, weight in zip(vocab_list, weights):
			if "\\" not in voc[0]:
				nb_tabs_ = (28. - len(voc[0])) / float(tab_value)
				nb_tabs = math.ceil(nb_tabs_)
				print(voc[0] + "\t" * nb_tabs + str(weight))
	else:
		print("Completion: {}/{}".format(weights.size - np.count_nonzero(weights), weights.size))


def main(datapath, base_w, lost_w, win_w):
	with open(datapath, "r") as f:
		list_vocab = [line_to_list_element(line) for line in f]

	# We also need to include characters:
	to_add = list(glob("./characters/*/*.png"))
	to_add_truth = [x.split("\\")[-1].split(".")[0].lower() for x in to_add]
	list_vocab += list(zip(to_add, to_add_truth))
	white_image = np.ones((500, 500, 4))
	white_image[:, :, 3] = 0

	# Each word has points that determine the probability of being chosen.
	weights = np.array([base_w for _ in list_vocab])

	np.random.seed(None)
	integers = list(range(len(weights)))

	plt.ion()
	plt.show()

	for _ in range(1000):
		print_scores(list_vocab, weights)
		print("\nYour current score is", get_score(weights), "\n")

		i = np.random.choice(integers, p=normalize(weights))
		current_word = list_vocab[i]

		if "\\" in current_word[0]:
			img = mpimg.imread(current_word[0])
			this_plot = plt.imshow(img)
			plt.pause(0.001)
			usr_input = input("What is this symbol? \n")
		else:
			usr_input = input("What's the japanese translation of \"" + current_word[0] + "\"?" + "\n")
			this_plot = None

		usr_input = usr_input.strip()

		if usr_input == current_word[1]:
			# Win
			weights[i] -= win_w
			print("Good answer!")
			if weights[i] <= 0:
				print("you've finished the word", current_word)
				weights[i] = 0
		else:
			# Loose
			weights[i] += lost_w
			print("The right answer was:", print_clear(usr_input, current_word[1]))
		input()
		if this_plot is not None:
			plt.clf()
			plt.imshow(white_image)
			plt.pause(0.001)
		cls()
		if get_score(weights) == 0:
			print("you won!")
			break


if __name__ == "__main__":
	main("./data.txt", 1.5, 1.5, 1)
