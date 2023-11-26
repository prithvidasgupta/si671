# PM Notes

Use HashingVectorizer to encode text for ML
- https://machinelearningmastery.com/prepare-text-data-machine-learning-scikit-learn/
- Use Bloom Filter to pick hash length?
- https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.HashingVectorizer.html
- https://stackoverflow.com/questions/21921987/why-is-my-scikit-learn-hashingvectorizor-giving-me-floats-with-binary-true-set

1. Extract categories from complaint descriptions
- Okay, this: How does this work with an LLM?
- Map NHSTA category to most general category
2. Track frequency of categories over time--do we see the matches fitting the known recall (COMPDESC == COMPNAME, Vehicle Make/Model == Vehicle Make/Model)?
- Spike analysis for certain makes/models
3. Are manufacturers addressing issues in a timely manner? (Jaccard similarity for descriptions between months--if similarity remains high, might indicated issues are going unaddressed)
- Break down by mfr, car make, car model
4. Correlation between particular component and severity of crash
5. Train ML to predict categories based on complaint descriptions
6. Compare performance of ML to LLM on category prediction

# Plan
- Spike analysis specifically on expert categorization of faulty parts
- Spike analysis specifically on LLM categorization on faulty parts
- If LLM has ~60% accuracy, then expert + LLM could maybe predict recalls before they happen
- Choose 10 categories (airbag, ignition)
    - Do count on component parts