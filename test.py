from response import generate_response

query = []
for line in open("query.txt"):
    query.append(line.strip())

output_file = open("output2.txt", "a")
for q in query:
    response, context_sources = generate_response(q, '')
    output_file.write(f"Question: {q}\n\nResponse: {response}\n\nContext Sources: {context_sources}\n\n\n")    

