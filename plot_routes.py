import matplotlib.pyplot as plt

def plot_routes(customers, routes, depot=None, distance_matrix=None, title=None):
    plt.figure(figsize=(10, 6))

    import matplotlib.cm as cm
    colors = cm.tab20.colors

    # 🔴 Depot
    if depot is None:
        depot = customers[0]

    plt.scatter(depot.x_coord, depot.y_coord, c='red', s=150, marker='s', label='Depot')

    # 🔵 Customers
    for i, c in enumerate(customers):
        if i == 0:
            continue
        plt.scatter(c.x_coord, c.y_coord, c='black', s=20)
        plt.text(c.x_coord, c.y_coord, str(i), fontsize=6)

    # 🚚 Routes
    for idx, route in enumerate(routes):
        if not route:
            continue

        color = colors[idx % len(colors)]

        x_coords = [depot.x_coord]
        y_coords = [depot.y_coord]

        distance = 0
        prev = 0

        for cid in route:
            customer = customers[cid]
            x_coords.append(customer.x_coord)
            y_coords.append(customer.y_coord)

            if distance_matrix is not None:
                distance += distance_matrix[prev][cid]

            prev = cid

        # quay về depot
        x_coords.append(depot.x_coord)
        y_coords.append(depot.y_coord)

        if distance_matrix is not None:
            distance += distance_matrix[prev][0]

        label = f"Xe {idx+1} (n={len(route)}"
        if distance_matrix is not None:
            label += f", d={round(distance,1)}"
        label += ")"

        plt.plot(x_coords, y_coords, color=color, linewidth=2, marker='o', label=label)

    # 📌 Title
    if title:
        plt.title(title, fontsize=14)
    else:
        plt.title("VRPTW Routes Visualization", fontsize=14)

    plt.xlabel("X")
    plt.ylabel("Y")
    plt.grid(alpha=0.3)

    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=7)

    plt.tight_layout()
    plt.show()