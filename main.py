import os.path
import pickle
import sys
import face_recognition


def train_model_by_image(path_dir, person_name):
    """Функция для обучения модели распознавания лиц при помощи директории с изображениями"""
    normalized_path_dir = path_dir.replace('"', '').replace('\\', '/')

    if not os.path.exists(normalized_path_dir):
        print('[ОШИБКА] Нет директории с таким названием!')
        sys.exit()

    suitable_persons = []
    images_for_train = os.listdir(normalized_path_dir)

    for (index, image) in enumerate(images_for_train):
        print(f'[ПРОГРЕСС] Обработано изображение {index + 1}/{len(images_for_train)}')

        img_to_numpy = face_recognition.load_image_file(f'{normalized_path_dir}/{image}')
        face_encoding = face_recognition.face_encodings(img_to_numpy)[0]

        if len(suitable_persons) == 0:
            suitable_persons.append(face_encoding)
        else:
            for i in range(len(suitable_persons)):
                result_compare = face_recognition.compare_faces([face_encoding], suitable_persons[i])

                if result_compare[0]:
                    suitable_persons.append(face_encoding)
                    print('[ИНФОРМАЦИЯ] Найдено совпадение!')
                    break

                else:
                    print('[ИНФОРМАЦИЯ] Возможно другой человек!')
                    break
        print()

    print(f'Длина списка подходящих лиц: {len(suitable_persons)}')

    result_data = {
        'name': person_name,
        'encodings_list': suitable_persons
    }

    if not os.path.exists('result_train'):
        os.mkdir('result_train')

    with open(f'result_train/{person_name}_encodings.pickle', 'wb') as file:
        file.write(pickle.dumps(result_data))

    return f'[ИНФОРМАЦИЯ] Процесс тренировки завершен, путь до файла: result_train/{person_name}_encodings.pickle'


def main():
    path_dir = input('Путь до директории с изображениями: ')
    person_name = input('Имя человека: ')
    print(train_model_by_image(path_dir=path_dir, person_name=person_name))


if __name__ == '__main__':
    main()