import os


def mkdir(p):
  if not os.path.exists(p):
    os.mkdir(p)


def link(src, dst):
  if not os.path.exists(dst):
    os.symlink(src, dst, target_is_directory=True)


if __name__ == '__main__':

    target_dir = '../data/fruits-360-small'

    mkdir(target_dir)

    classes = [
      'Apple Golden 1',
      'Avocado',
      'Lemon',
      'Mango',
      'Kiwi',
      'Banana',
      'Strawberry',
      'Raspberry'
    ]

    train_path_from = os.path.abspath('../data/fruits-360/Training')
    test_path_from = os.path.abspath('../data/fruits-360/Test')

    train_path_to = os.path.abspath(os.path.join(target_dir, 'Training'))
    test_path_to = os.path.abspath(os.path.join(target_dir, 'Test'))

    mkdir(train_path_to)
    mkdir(test_path_to)

    for c in classes:
      link(train_path_from + '/' + c, train_path_to + '/' + c)
      link(test_path_from + '/' + c, test_path_to + '/' + c)