
import os
import random
import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk
from PIL import Image, ImageTk

# Tymczasowe funkcje do detekcji

def general_detection(image_path):
    # Funkcja do symulacji ogólnej detekcji na obrazie
    print(f"Ogólna detekcja dla obrazu: {image_path}")  # Debug log
    messagebox.showinfo("Ogólna Detekcja", f"Wykonano ogólną detekcję na obrazie: {image_path}")

def gender_detection(image_path):
    # Funkcja do symulacji detekcji płci na obrazie
    print(f"Detekcja płci dla obrazu: {image_path}")  # Debug log
    messagebox.showinfo("Detekcja Płci", f"Wykonano detekcję płci na obrazie: {image_path}")

def age_detection(image_path):
    # Funkcja do symulacji detekcji wieku na obrazie
    print(f"Detekcja wieku dla obrazu: {image_path}")  # Debug log
    messagebox.showinfo("Detekcja Wieku", f"Wykonano detekcję wieku na obrazie: {image_path}")

# Główna klasa aplikacji
class PhotoAnalysisApp:
    def __init__(self, root):
        # Konstruktor inicjalizujący główne elementy GUI
        print("Inicjalizacja aplikacji...")  # Debug log
        self.root = root
        self.root.title("Aplikacja do Analizy Zdjęć")
        self.root.geometry("800x600")  # Ustawienie rozmiaru okna

        # Główna ramka
        self.main_frame = ttk.Frame(self.root)
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Ramka do wyświetlania obrazu, umieszczona po lewej stronie okna
        self.image_frame = ttk.Frame(self.main_frame)
        self.image_frame.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        self.image_frame.columnconfigure(0, weight=1)
        self.image_frame.rowconfigure(0, weight=1)

        # Etykieta obrazu - początkowo wyświetla komunikat o braku załadowanego obrazu
        self.image_label = ttk.Label(self.image_frame, text="Brak załadowanego obrazu", relief=tk.SUNKEN, anchor="center")
        self.image_label.grid(row=0, column=0, sticky="nsew")

        # Przycisk do wczytania losowego obrazu
        self.load_button = ttk.Button(self.image_frame, text="Wczytaj Losowy Obraz", command=self.load_random_image)
        self.load_button.grid(row=1, column=0, pady=10)

        # Ramka na przyciski, umieszczona po prawej stronie okna
        self.button_frame = ttk.Frame(self.main_frame)
        self.button_frame.grid(row=0, column=1, sticky="ns", padx=10, pady=10)

        # Przyciski do różnych detekcji - ogólna, płci i wieku
        self.general_button = ttk.Button(self.button_frame, text="Ogólna Detekcja", command=self.general_detection)
        self.general_button.pack(pady=5, fill=tk.X)

        self.gender_button = ttk.Button(self.button_frame, text="Detekcja Płci", command=self.gender_detection)
        self.gender_button.pack(pady=5, fill=tk.X)

        self.age_button = ttk.Button(self.button_frame, text="Detekcja Wieku", command=self.age_detection)
        self.age_button.pack(pady=5, fill=tk.X)

        # Ramka na opis
        self.description_frame = ttk.Frame(self.main_frame)
        self.description_frame.grid(row=1, column=0, columnspan=2, sticky="ew", padx=10, pady=10)

        # Pole tekstowe na opis (4 linie wolne)
        self.description_label = ttk.Label(self.description_frame, text="Opis: (4 linie wolne)", anchor="w")
        self.description_label.pack(fill=tk.X, padx=5, pady=5)

        # Ścieżka do folderu ze zdjęciami - użytkownik zostanie poproszony o jej wskazanie
        print("Prośba o wybranie folderu ze zdjęciami...")  # Debug log
        self.images_path = filedialog.askdirectory(title="Wybierz Folder ze Zdjęciami")
        self.current_image_path = None

    def load_random_image(self):
        # Funkcja do wczytania losowego obrazu z wybranego folderu
        print("Próba wczytania losowego obrazu...")  # Debug log
        if not self.images_path:
            # Jeśli użytkownik nie wybrał folderu, wyświetl ostrzeżenie
            print("Błąd: Nie wybrano folderu ze zdjęciami.")  # Debug log
            messagebox.showwarning("Błąd", "Nie wybrano folderu ze zdjęciami.")
            return

        # Filtruj tylko pliki graficzne (png, jpg, jpeg) z wybranego folderu
        image_files = [f for f in os.listdir(self.images_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        print(f"Znaleziono {len(image_files)} plików graficznych.")  # Debug log
        if not image_files:
            # Jeśli w folderze nie ma żadnych plików graficznych, wyświetl ostrzeżenie
            print("Błąd: Brak obrazów w wybranym folderze.")  # Debug log
            messagebox.showwarning("Błąd", "Brak obrazów w wybranym folderze.")
            return

        # Wybierz losowy obraz z listy dostępnych plików graficznych
        random_image = random.choice(image_files)
        print(f"Wybrano losowy obraz: {random_image}")  # Debug log
        self.current_image_path = os.path.join(self.images_path, random_image)
        self.display_image(self.current_image_path)

    def display_image(self, image_path):
        # Funkcja do wyświetlania wybranego obrazu w GUI
        print(f"Wyświetlanie obrazu: {image_path}")  # Debug log
        try:
            image = Image.open(image_path)  # Otwórz obraz za pomocą PIL
            image = image.resize((400, 400), Image.ANTIALIAS)  # Zmień rozmiar obrazu do 400x400 pikseli
            photo = ImageTk.PhotoImage(image)  # Przekształć obraz do formatu obsługiwanego przez tkinter
            self.image_label.config(image=photo, text="")  # Ustaw obraz w etykiecie, usuń tekst
            self.image_label.image = photo  # Zachowaj odniesienie do obiektu obrazu, aby nie został usunięty przez garbage collector
        except Exception as e:
            # Jeśli nie uda się załadować obrazu, wyświetl komunikat o błędzie
            print(f"Błąd podczas ładowania obrazu: {e}")  # Debug log
            messagebox.showerror("Błąd", f"Nie udało się załadować obrazu: {e}")

    def general_detection(self):
        # Funkcja obsługująca przycisk ogólnej detekcji
        print("Rozpoczęcie ogólnej detekcji...")  # Debug log
        if self.current_image_path:
            # Jeśli obraz jest załadowany, wykonaj detekcję
            general_detection(self.current_image_path)
        else:
            # Jeśli nie ma obrazu, wyświetl ostrzeżenie
            print("Błąd: Brak wybranego obrazu do detekcji.")  # Debug log
            messagebox.showwarning("Błąd", "Brak wybranego obrazu.")

    def gender_detection(self):
        # Funkcja obsługująca przycisk detekcji płci
        print("Rozpoczęcie detekcji płci...")  # Debug log
        if self.current_image_path:
            # Jeśli obraz jest załadowany, wykonaj detekcję
            gender_detection(self.current_image_path)
        else:
            # Jeśli nie ma obrazu, wyświetl ostrzeżenie
            print("Błąd: Brak wybranego obrazu do detekcji.")  # Debug log
            messagebox.showwarning("Błąd", "Brak wybranego obrazu.")

    def age_detection(self):
        # Funkcja obsługująca przycisk detekcji wieku
        print("Rozpoczęcie detekcji wieku...")  # Debug log
        if self.current_image_path:
            # Jeśli obraz jest załadowany, wykonaj detekcję
            age_detection(self.current_image_path)
        else:
            # Jeśli nie ma obrazu, wyświetl ostrzeżenie
            print("Błąd: Brak wybranego obrazu do detekcji.")  # Debug log
            messagebox.showwarning("Błąd", "Brak wybranego obrazu.")

# Główny skrypt do uruchomienia aplikacji
if __name__ == "__main__":
    # Tworzenie głównego okna aplikacji i uruchomienie pętli zdarzeń
    print("Uruchamianie aplikacji...")  # Debug log
    root = tk.Tk()
    app = PhotoAnalysisApp(root)
    root.mainloop()
