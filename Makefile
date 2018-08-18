
requirements.txt:
	pipenv lock --requirements > requirements.txt

.PHONY: clean
clean:
	rm requirements.txt
