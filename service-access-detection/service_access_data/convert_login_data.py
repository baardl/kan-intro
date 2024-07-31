# Åpne den opprinnelige filen i lesemodus
with open('login_data.csv', 'r') as original_file:
    lines = original_file.readlines()

    # Skriv ut hver linje til konsollen
    for line in lines:
        print(line, end='')

    # Åpne en ny fil i skrivemodus
    with open('login_data_copy.csv', 'w') as new_file:
        # Skriv hver linje til den nye filen
        for line in lines:
            new_file.write(line)