**Chatbot**

Creation of a retrieval-based chatbot using predefined input patterns and responses.

The chatbot is trained on a dataset which contains categories (intents), patterns and responses. 
We used a  recurrent neural network (RNN) to classify which category the user's message belongs to and then,we gave a random response from the list of responses that we created.


**The dataset**

We created the ‘intents.json’. It is a JSON file that contains the patterns we need to find, and the responses we want to return to a user's question.

**project files**

    Intents.json – The data file which has predefined patterns and responses

    train_chatbot.py – In this Python file, we wrote a script to build the model and train our chatbot.

    Words.pkl – This is a pickle file in which we store the words Python object that contains a list of our vocabulary.

    Classes.pkl – The classes pickle file contains the list of categories.

    model.h5 – This is the trained model.

    main.py – This is the Python script in which we developed a GUI for our chatbot. Users are able to interact with the bot.
    
    requirements.txt - This is a text file which contains the prerequisites packages
