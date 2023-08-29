import matplotlib.pyplot as plt

def main():
    with open('memory_used.txt', 'r') as f:
        lines = f.readlines()
    lines = [float(line[:-1]) for line in lines]

    # plt.figure(figsize=(20,5))
    plt.plot(lines, color='r')
    plt.title('System Memory')
    plt.xlabel('Steps')
    plt.ylabel('Memory Used (GB)')
    plt.ylim([15, 21])
    plt.savefig('memory_used.png')

if __name__ == '__main__':
    main()