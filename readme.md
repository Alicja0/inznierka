# Eye disease classification

Propozycja 
Wykrywanie jaskru na podstawie obrazów dan oka jest kluczowe dla zapobiegania poważnej utracie wzroku. Prejket ten ma na celu storzenie systemu, który automtyzuje klasyfikację obrazy dan oka pod kontem obecności jaskry.

## Problem description
Ta sekcja opisuje zbiór danych użytych w projekcie. Zawiera szczegóły na temat zródła danych, rodzaju obrazów ( obrazy dna oka), struktury zbioru dancyh oraz wszelkich kroków przetwarzania wstępnego, które zostały podjete przed trenowaniem modelu

## Data description
Zbiór danych użyty w projekcie pochodzi z bazy ODIR-5K i zawiera obrazy dan oka.Dane zostały wstęnie przetworzone, aby usunąć tło i zmniejszyć szumy, co zwiększa dokładność modelu.

## Setup
### Local
Instrukcje dotyczące konfiguracji projektu na lokalnej maszynie .
-Wymagania wstępne ( wersja Pythona, wymagane biblioteki)
- Kroki, aby sklonować repo
- Instrukcje dotyczące konfiguracji wirtualnego środowiska
- polecenia do zasintalowania zalezności 
- jak uruchomic projekt lokalnie

### Colab
Instrukcje dotyczące konfiguracji i uruchomienia projekty w Google Colab. 
- jak otowrzyć projekt w Colab
- instalowanie wymaganych bibliotek
- przesyłanie zbioru dancyh do Colab
- uruchamianie kodu w nootebooku Colab

## Model
Ta sekcja zawiera szczegóły dotyczące architektury modelu używanego do wykrywania chorów oczu. Obejmuje to:
- Model bazowy ( np ConyNeXtTiny)
- Szczegóły dotyczące procesu treningowego ( np funkcja straty, metryki, liczba epok )
- jak interpretować wyniki modelu
## Results 
Ta sekcja zawiera wyniki uzyskane z modelu. 
- Dokładność modelu na zbiorze testowym
- instone metryki (np. precyzja, recall, f1-score)
- Wykresy przedstawiające wyniki modelu
- Przykłady poprawnych i błędncyh klasyfikacji 
## Usage
Instrukcje, jak używać modelu do klasyfikacji nowych obrazów;
- jak przygotować nowe obrazy 
- jak załadować wytrenowany model
- jak uruchomić predykcję na nowych obrazach
- jak interpretować wyniki predykcji
## Future Work
Sugestie dotyczące przyszłych prac i możliwych ulepszeń projektu:
- możliwość poprawy modelu
- rozszerzenie projektu na inne choroby oczu
- integracja z systemami klinicznymi 
- dalsze badania i experymenty
  
## Contributors
- Alicja Polewska 
- Artur Żokowski 
