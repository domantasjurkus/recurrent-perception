import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

all_categories = ['pant', 'shirt', 'sweater', 'tshirt']

confusion = np.array([[ 196.0,   30,   48,   40],
        [  45,  149,   55,   72],
        [  49,   22,  163,   51],
        [  65,   56,   71,  154]])

def confusion_matrix(confusion):
    # Normalize by dividing every row by its sum
    for i in range(len(all_categories)):
        confusion[i] = confusion[i] / confusion[i].sum()

    # Set up plot
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(confusion)
    fig.colorbar(cax)
    plt.xlabel('prediction', labelpad=15)
    plt.ylabel('actual', labelpad=15)

    # Set up axes
    ax.set_xticklabels([''] + all_categories)
    ax.set_yticklabels([''] + all_categories)

    # Force label at every tick
    # ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    # ax.yaxis.set_major_locator(ticker.MultipleLocator(1))


    for (i, j), z in np.ndenumerate(confusion):
        ax.text(j, i, '{:0.1f}'.format(z), ha='center', va='center',
                bbox=dict(boxstyle='round', facecolor='white', edgecolor='0.3'))

    # sphinx_gallery_thumbnail_number = 2
    plt.show()

confusion_matrix(confusion)