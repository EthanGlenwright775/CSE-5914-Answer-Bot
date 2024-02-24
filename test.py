#import the method from the .py file from the connections folder
from connections.rephraser import rephrase
from connections.q_generator import generate_questions_multicontext
from connections.q_generator import generate_questions_monocontext

#Using the rephraser
to_be_rephrased = "We should be able to import functions from any one of these scripts to any other. As we'll see below, importing from child directories is quite straightforward whereas importing from parent or sibling directories is more complex."
is_now_rephrased = rephrase(to_be_rephrased)

print(is_now_rephrased)
print("")


#Using the quesiton generator (multicontext)
answers1 = ["most widely grown species in the genus Malus", "several kinds of large herbaceous flowering plants in the genus Musa", "chemical formula H 2O"]
context1 = ["An apple is a round, edible fruit produced by an apple tree Apple trees are cultivated worldwide and are the most widely grown species in the genus Malus.",
            "A banana is an elongated, edible fruit – botanically a berry – produced by several kinds of large herbaceous flowering plants in the genus Musa.",
            "Water is an inorganic compound with the chemical formula H 2O. It is a transparent, tasteless, odorless, and nearly colorless chemical substance."]
output1 = generate_questions_multicontext(answers1, context1)

for question in output1: print(question)
print("")


#Using the monocontext variant (note the context is no longer a list)
answers2 = ["sequence-to-sequence question generator", "generates a question as an output"]
context2 = "This model is a sequence-to-sequence question generator which takes an answer and context as an input, and generates a question as an output. It is based on a pretrained t5-base model."
output2 = generate_questions_monocontext(answers2, context2)

for question in output2: print(question)