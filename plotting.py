import matplotlib.pyplot as plt

def classes_colored_scatter(x, y, c, data, cmap = 'cividis', loc = 'lower right', title = 'Classes'):
    fig, ax = plt.subplots(figsize = (8, 5))
    scatter = ax.scatter(x = x, y = y, c = c, data = data, cmap = cmap)
    
    plt.xlabel(x)
    plt.ylabel(y)
    plt.box(on = None)
    legend1 = ax.legend(*scatter.legend_elements(), loc = loc, title = title)
    ax.add_artist(legend1)