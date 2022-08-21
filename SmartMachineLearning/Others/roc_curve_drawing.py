import matplotlib.pyplot as plt

class roc_curve_drawing:
    def __init__(self):
        pass
    
    def draw_curve(self, fprs, tprs, labels = None, colors = None, linestyles = None, linewidths = None,
                   title = None, legend_pos = None, num_class = None):
        for x in range(num_class):
            plt.plot(fprs[x], tprs[x], 
                     color = colors[x] if colors else None, 
                     label = labels[x] if labels else None, 
                     linestyle = linestyles[x] if linestyles else None, 
                     linewidth = linewidths[x] if linewidths else None)
        
        plt.plot([0, 1], [0, 1], "k--", lw=2)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(title)
        plt.legend(loc=legend_pos)
        plt.show()