import matplotlib.pyplot as plt
import pickle

class Graphics:
    def __init__(self):
        self.current_somme_loss_estimation = 0
        self.current_somme_ecart = 0
        self.current_somme_absolute_distance = 0
        self.current_somme_tau_hard_score = 0
        
        self.current_somme_loss_estimation_checkmate = 0
        self.current_somme_ecart_checkmate = 0
        self.current_somme_absolute_distance_checkmate = 0

        self.current_num_iteration = 0

        self.liste_mean_loss_estimation = []
        self.liste_mean_ecart = []
        self.liste_mean_absolute_distance = []
        self.liste_tau_hard_score = []

        self.liste_mean_loss_estimation_checkmate = []
        self.liste_mean_ecart_checkmate = []
        self.liste_mean_absolute_distance_checkmate = []

    def save(self, filename):
        """
        Save the current instance of Graphics to a file using pickle.
        :param filename: Path to the file where the object will be saved.
        """
        with open(filename, 'wb') as file:
            pickle.dump(self, file)

    @staticmethod
    def reload(filename):
        """
        Reload a Graphics instance from a file using pickle.
        :param filename: Path to the file from which the object will be reloaded.
        :return: The reloaded Graphics instance.
        """
        with open(filename, 'rb') as file:
            obj = pickle.load(file)
        print(f"Graphics object reloaded from {filename}")
        return obj
    
    def add(self, loss_estimation, ecart, absolute_distance, tau_hard_score, 
            loss_estimation_checkmate, ecart_checkmate, absolute_distance_checkmate):
        self.current_somme_loss_estimation += loss_estimation
        self.current_somme_ecart += ecart
        self.current_somme_absolute_distance += absolute_distance
        self.current_somme_tau_hard_score += tau_hard_score

        self.current_somme_loss_estimation_checkmate += loss_estimation_checkmate
        self.current_somme_ecart_checkmate += ecart_checkmate
        self.current_somme_absolute_distance_checkmate += absolute_distance_checkmate

        self.current_num_iteration += 1

    def push(self):
        if self.current_num_iteration == 0:
            print("######################\n" * 10 + "\n\ncurrent_iteration nul dans graphics.push ! \n\n" + "######################\n" * 10)
        else:
            self.liste_mean_loss_estimation.append(self.current_somme_loss_estimation / self.current_num_iteration)
            self.liste_mean_absolute_distance.append(self.current_somme_absolute_distance / self.current_num_iteration)
            self.liste_mean_ecart.append(self.current_somme_ecart / self.current_num_iteration)
            self.liste_tau_hard_score.append(self.current_somme_tau_hard_score / self.current_num_iteration)

            self.liste_mean_loss_estimation_checkmate.append(self.current_somme_loss_estimation_checkmate / self.current_num_iteration)
            self.liste_mean_ecart_checkmate.append(self.current_somme_ecart_checkmate / self.current_num_iteration)
            self.liste_mean_absolute_distance_checkmate.append(self.current_somme_absolute_distance_checkmate / self.current_num_iteration)

            self.current_somme_loss_estimation = 0
            self.current_somme_ecart = 0
            self.current_somme_absolute_distance = 0
            self.current_somme_tau_hard_score = 0

            self.current_somme_loss_estimation_checkmate = 0
            self.current_somme_ecart_checkmate = 0
            self.current_somme_absolute_distance_checkmate = 0
            
            self.current_num_iteration = 0

    def save_plot(self, filename):
        """
        Sauvegarde le tracé des courbes sur une seule figure avec 5 sous-graphiques (subplots).
        :param filename: Nom du fichier de sortie (par exemple, 'plot.png')
        """
        plt.figure(figsize=(12, 20))

        # Graphique 1 : Loss Estimation
        plt.subplot(5, 1, 1)
        plt.plot(self.liste_mean_loss_estimation, label="Mean Loss Estimation")
        plt.yscale('log')
        plt.title("Mean Loss Estimation (Logarithmic Scale)")
        plt.xlabel("Iterations")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True)

        # Graphique 2 : Loss Estimation Checkmate
        plt.subplot(5, 1, 2)
        plt.plot(self.liste_mean_loss_estimation_checkmate, label="Mean Loss Estimation Checkmate", color="red")
        plt.yscale('log')
        plt.title("Mean Loss Estimation Checkmate (Logarithmic Scale)")
        plt.xlabel("Iterations")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True)

        # Graphique 3 : Distance et Ecart
        plt.subplot(5, 1, 3)
        plt.plot(self.liste_mean_absolute_distance, label="Mean Absolute Distance", color="green")
        plt.plot(self.liste_mean_ecart, label="Mean Ecart", color="orange")
        plt.title("Mean Absolute Distance and Ecart")
        plt.xlabel("Iterations")
        plt.ylabel("Values")
        plt.legend()
        plt.grid(True)

        # Graphique 4 : Distance Checkmate et Ecart Checkmate
        plt.subplot(5, 1, 4)
        plt.plot(self.liste_mean_absolute_distance_checkmate, label="Mean Absolute Distance Checkmate", color="purple")
        plt.plot(self.liste_mean_ecart_checkmate, label="Mean Ecart Checkmate", color="red")
        plt.title("Mean Absolute Distance Checkmate and Ecart Checkmate")
        plt.xlabel("Iterations")
        plt.ylabel("Values")
        plt.legend()
        plt.grid(True)

        # Graphique 5 : Tau Hard Score
        plt.subplot(5, 1, 5)
        plt.plot(self.liste_tau_hard_score, label="Tau Hard Score", color="blue")
        plt.title("Tau Hard Score")
        plt.xlabel("Iterations")
        plt.ylabel("Score")
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.savefig(filename)
        plt.close()
