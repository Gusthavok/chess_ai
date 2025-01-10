import matplotlib.pyplot as plt
import pickle

class Graphics:
    def __init__(self):
        self.current_somme_loss_estimation = 0
        self.current_somme_ecart = 0
        self.current_somme_absolute_distance = 0
        self.current_somme_tau_hard_score = 0
        
        self.current_num_iteration = 0

        self.liste_mean_loss_estimation = []
        self.liste_mean_ecart = []
        self.liste_mean_absolute_distance = []
        self.liste_tau_hard_score = []

    def save(self, filename):
        """
        Save the current instance of Graphics to a file using pickle.
        :param filename: Path to the file where the object will be saved.
        """
        with open(filename, 'wb') as file:
            pickle.dump(self, file)
        print(f"Graphics object saved to {filename}")

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
    
    def add(self, loss_estimation, ecart, absolute_distance, tau_hard_score):
        self.current_somme_loss_estimation += loss_estimation
        self.current_somme_ecart += ecart
        self.current_somme_absolute_distance += absolute_distance
        self.current_somme_tau_hard_score += tau_hard_score
        self.current_num_iteration += 1

    def push(self):
        if self.current_num_iteration == 0:
            print("######################\n" * 10 + "\n\ncurrent_iteration nul dans graphics.push ! \n\n" + "######################\n" * 10)
        else:
            self.liste_mean_loss_estimation.append(self.current_somme_loss_estimation / self.current_num_iteration)
            self.liste_mean_absolute_distance.append(self.current_somme_absolute_distance / self.current_num_iteration)
            self.liste_mean_ecart.append(self.current_somme_ecart / self.current_num_iteration)
            self.liste_tau_hard_score.append(self.current_somme_tau_hard_score / self.current_num_iteration)
            
            self.current_somme_loss_estimation = 0
            self.current_somme_ecart = 0
            self.current_somme_absolute_distance = 0
            self.current_somme_tau_hard_score = 0            
            self.current_num_iteration = 0

    def save_plot(self, filename):
        """
        Sauvegarde le tracé des courbes dans un fichier avec un axe Y logarithmique pour la loss.
        :param filename: Nom du fichier de sortie (par exemple, 'plot.png')
        """
        plt.figure(figsize=(10, 6))

        # Tracé de la loss estimation avec un axe Y logarithmique
        plt.subplot(3, 1, 1)
        plt.plot(self.liste_mean_loss_estimation, label="Mean Loss Estimation")
        plt.yscale('log')
        plt.title("Mean Loss Estimation (Logarithmic Scale)")
        plt.xlabel("Iterations")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True)

        # Tracé des autres métriques
        plt.subplot(3, 1, 2)
        plt.plot(self.liste_mean_ecart, label="Mean Ecart", color="orange")
        plt.plot(self.liste_mean_absolute_distance, label="Mean Absolute Distance", color="green")
        plt.title("Mean Ecart and Absolute Distance")
        plt.xlabel("Iterations")
        plt.ylim((0, 5))
        plt.ylabel("Values")
        plt.legend()
        plt.grid(True)
        
        plt.subplot(3, 1, 3)
        plt.plot(self.liste_tau_hard_score, label="tau hard score", color="green")
        plt.title("Taux hard score through iterations")
        plt.xlabel("Iterations")
        plt.ylabel("Values")
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.savefig(filename)
        plt.close()
