import json
import csv


def decode(values):
    res = []
    for value in values:
        res.append(value.replace('\u2013', '-'))
    return res


results_input = open('results.json', 'r')
results_output = open('results.csv', 'w')

estimates = json.load(results_input)
csv_writer = csv.writer(results_output, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

# write header, remove if not needed
first_estimate = estimates[0]
header = first_estimate.keys()
csv_writer.writerow(header)

for estimate in estimates:
    csv_writer.writerow(decode(estimate.values()))

results_input.close()
results_output.close()

