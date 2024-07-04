import replicate
import streamlit as st
import requests
import zipfile
import io
from utils import icon
from streamlit_image_select import image_select

# Настройки интерфейса
st.set_page_config(page_title="Генератор копий изображений",
                   page_icon=":bridge_at_night:",
                   layout="wide")
icon.show_icon(":foggy:")
st.markdown("# :rainbow[Text-to-Image Artistry Studio]")

# Токены API и конечные точки из файла `.streamlit/secrets.toml`
REPLICATE_API_TOKEN = st.secrets["r8_W3gxzPKhTzULGXKQfHXVMIyFRkqr7634AzmTI"]
REPLICATE_MODEL_ENDPOINTSTABILITY = st.secrets["REPLICATE_MODEL_ENDPOINTSTABILITY"]

# Текст ресурсов, ссылка и логотип
replicate_text = "Стабильность модели AI SDXL на копиях"
replicate_link = "https://replicate.com/stability-ai/sdxl"
replicate_logo = "https://storage.googleapis.com/llama2_release/Screen%20Shot%202023-07-21%20at%2012.34.05%20PM.png"

# Заполнители для изображений и галереи
generated_images_placeholder = st.empty()
gallery_placeholder = st.empty()

def configure_sidebar() -> None:
    """
    Настройка и отображение элементов боковой панели.

    Эта функция настраивает боковую панель приложения Streamlit,
    включая форму для ввода данных пользователем и раздел ресурсов.
    """
    with st.sidebar:
        with st.form("my_form"):
            st.info("**Привет, начинайте здесь ↓**", icon="👋🏾")
            with st.expander(":rainbow[**Уточните результат здесь**]"):
                # Расширенные настройки (для любопытных!)
                width = st.number_input("Ширина выходного изображения", value=1024)
                height = st.number_input("Высота выходного изображения", value=1024)
                num_outputs = st.slider(
                    "Количество выходных изображений", value=1, min_value=1, max_value=4)
                scheduler = st.selectbox('Планировщик', ('DDIM', 'DPMSolverMultistep', 'HeunDiscrete',
                                                         'KarrasDPM', 'K_EULER_ANCESTRAL', 'K_EULER', 'PNDM'))
                num_inference_steps = st.slider(
                    "Количество шагов денойзинга", value=50, min_value=1, max_value=500)
                guidance_scale = st.slider(
                    "Масштаб для классификатор-независимого руководства", value=7.5, min_value=1.0, max_value=50.0, step=0.1)
                prompt_strength = st.slider(
                    "Сила подсказки при использовании img2img/inpaint (1.0 соответствует полной деструкции информации в изображении)", value=0.8, max_value=1.0, step=0.1)
                refine = st.selectbox(
                    "Выберите стиль уточнения (остались только 2)", ("expert_ensemble_refiner", "None"))
                high_noise_frac = st.slider(
                    "Доля шума для `expert_ensemble_refiner`", value=0.8, max_value=1.0, step=0.1)
            prompt = st.text_area(
                ":orange[**Введите подсказку: начните печатать, Шекспир ✍🏾**]",
                value="Космонавт, едущий на радужном единороге, кинематографично, драматично")
            negative_prompt = st.text_area(":orange[**Что не хотите видеть на изображении? 🙅🏽‍♂️**]",
                                           value="ужаснейшее качество, искаженные черты",
                                           help="Это отрицательная подсказка, укажите, что не хотите видеть на созданном изображении")

            # Большая красная кнопка "Отправить"!
            submitted = st.form_submit_button(
                "Отправить", type="primary", use_container_width=True)

        # Кредиты и ресурсы
        st.divider()
        st.markdown(
            ":orange[**Ресурсы:**]  \n"
            f"<img src='{replicate_logo}' style='height: 1em'> [{replicate_text}]({replicate_link})",
            unsafe_allow_html=True
        )
        st.markdown(
            """
            ---
            Следите за мной:

            𝕏 → [@tonykipkemboi](https://twitter.com/tonykipkemboi)

            LinkedIn → [Tony Kipkemboi](https://www.linkedin.com/in/tonykipkemboi)

            """
        )

        return submitted, width, height, num_outputs, scheduler, num_inference_steps, guidance_scale, prompt_strength, refine, high_noise_frac, prompt, negative_prompt

def main_page(submitted: bool, width: int, height: int, num_outputs: int,
              scheduler: str, num_inference_steps: int, guidance_scale: float,
              prompt_strength: float, refine: str, high_noise_frac: float,
              prompt: str, negative_prompt: str) -> None:
    """Макет главной страницы и логика создания изображений.

    Args:
        submitted (bool): Флаг, указывающий, была ли отправлена форма.
        width (int): Ширина выходного изображения.
        height (int): Высота выходного изображения.
        num_outputs (int): Количество выходных изображений.
        scheduler (str): Тип планировщика для модели.
        num_inference_steps (int): Количество шагов денойзинга.
        guidance_scale (float): Масштаб для классификатор-независимого руководства.
        prompt_strength (float): Сила подсказки при использовании img2img/inpaint.
        refine (str): Стиль уточнения для использования.
        high_noise_frac (float): Доля шума для `expert_ensemble_refiner`.
        prompt (str): Текстовая подсказка для создания изображения.
        negative_prompt (str): Текстовая подсказка для элементов, которых нужно избегать в изображении.
    """
    if submitted:
        with st.status('👩🏾‍🍳 Создаем ваше искусство из слов...', expanded=True) as status:
            st.write("⚙️ Модель запущена")
            st.write("🙆‍♀️ Потянитесь пока что")
            try:
                # Вызов API только при нажатии кнопки "Отправить"
                if submitted:
                    # Вызов API replicate для получения изображения
                    with generated_images_placeholder.container():
                        all_images = []  # Список для хранения всех созданных изображений
                        output = replicate.run(
                            REPLICATE_MODEL_ENDPOINTSTABILITY,
                            input={
                                "prompt": prompt,
                                "width": width,
                                "height": height,
                                "num_outputs": num_outputs,
                                "scheduler": scheduler,
                                "num_inference_steps": num_inference_steps,
                                "guidance_scale": guidance_scale,
                                "prompt_stregth": prompt_strength,
                                "refine": refine,
                                "high_noise_frac": high_noise_frac
                            }
                        )
                        if output:
                            st.toast(
                                'Ваше изображение создано!', icon='😍')
                            # Сохранение созданного изображения в состоянии сессии
                            st.session_state.generated_image = output

                            # Отображение изображения
                            for image in st.session_state.generated_image:
                                with st.container():
                                    st.image(image, caption="Созданное изображение 🎈",
                                             use_column_width=True)
                                    # Добавление изображения в список
                                    all_images.append(image)

                                    response = requests.get(image)
                        # Сохранение всех созданных изображений в состоянии сессии
                        st.session_state.all_images = all_images

                        # Создание объекта BytesIO
                        zip_io = io.BytesIO()

                        # Опция скачивания для каждого изображения
                        with zipfile.ZipFile(zip_io, 'w') as zipf:
                            for i, image in enumerate(st.session_state.all_images):
                                response = requests.get(image)
                                if response.status_code == 200:
                                    image_data = response.content
                                    # Запись каждого изображения в zip файл с именем
                                    zipf.writestr(
                                        f"output_file_{i+1}.png", image_data)
                                else:
                                    st.error(
                                        f"Не удалось получить изображение {i+1} с {image}. Код ошибки: {response.status_code}", icon="🚨")
                        # Создание кнопки скачивания для zip файла
                        st.download_button(
                            ":red[**Скачать все изображения**]", data=zip_io.getvalue(), file_name="output_files.zip", mime="application/zip", use_container_width=True)
                status.update(label="✅ Изображения созданы!",
                              state="complete", expanded=False)
            except Exception as e:
                print(e)
                st.error(f'Возникла ошибка: {e}', icon="🚨")

    # Если не отправлено, остаемся здесь 🍹
    else:
        pass

    # Отображение галереи для вдохновения
    with gallery_placeholder.container():
        img = image_select(
            label="Нравится то, что видите? Сохраните изображение, если мы его поделились! 😉",
            images=[
                "gallery/farmer_sunset.png", "gallery/astro_on_unicorn.png",
                "gallery/friends.png", "gallery/wizard.png", "gallery/puppy.png",
                "gallery/cheetah.png", "gallery/viking.png",
            ],
            captions=["Фермер, работающий на тракторе на закате, кинематографично, драматично",
                      "Космонавт, едущий на радужном единороге, кинематографично, драматично",
                      "Группа друзей, смеющихся и танцующих на музыкальном фестивале, радостная атмосфера, фотография на пленку 35 мм",
                      "Волшебник, произносящий заклинание, из его рук исходит интенсивная магическая энергия, крайне детализированная фантастическая иллюстрация",
                      "Милый щенок, играющий на поле с цветами, малая глубина резкости, фотография на Canon",
                      "Мать-гепард кормит своих детенышей в высокой траве Серенгети. Утреннее солнце проникает через траву. Фотография National Geographic от Франса Лантина",
                      "Крупный портрет бородатого викинга в рогатом шлеме. Он смотрит вдаль, держа боевой топор. Драматическое освещение, цифровая масляная живопись",
                      ],
            use_container_width=True
        )

def main():
    """
    Основная функция для запуска приложения Streamlit.

    Эта функция инициализирует конфигурацию боковой панели и макет главной страницы.
    Она получает пользовательские данные с боковой панели и передает их в основную функцию страницы.
    Основная функция страницы затем создает изображения на основе этих данных.
    """
    submitted, width, height, num_outputs, scheduler, num_inference_steps, guidance_scale, prompt_strength, refine, high_noise_frac, prompt, negative_prompt = configure_sidebar()
    main_page(submitted, width, height, num_outputs, scheduler, num_inference_steps,
              guidance_scale, prompt_strength, refine, high_noise_frac, prompt, negative_prompt)

if __name__ == "__main__":
    main()
