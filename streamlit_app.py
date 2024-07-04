import replicate
import streamlit as st
import requests
import zipfile
import io
from utils import icon
from streamlit_image_select import image_select

# UI configurations
st.set_page_config(page_title="Генератор изображений Replicate",
                   page_icon=":bridge_at_night:",
                   layout="wide")
icon.show_icon(":foggy:")
st.markdown("# :rainbow[Студия Текст-Изображение]")

# API Tokens and endpoints from `.streamlit/secrets.toml` file
REPLICATE_API_TOKEN = st.secrets["REPLICATE_API_TOKEN"]
REPLICATE_MODEL_ENDPOINTSTABILITY = st.secrets["REPLICATE_MODEL_ENDPOINTSTABILITY"]

# Resources text, link, and logo
replicate_text = "Модель Stability AI SDXL на Replicate"
replicate_link = "https://replicate.com/stability-ai/sdxl"
replicate_logo = "friends.png"

# Placeholders for images and gallery
generated_images_placeholder = st.empty()
gallery_placeholder = st.empty()


def configure_sidebar() -> None:
    """
    Настройка и отображение элементов боковой панели.

    Эта функция настраивает боковую панель приложения Streamlit, 
    включая форму для ввода пользователем и раздел ресурсов.
    """
    with st.sidebar:
        with st.form("my_form"):
            st.info("**Начните здесь ↓**", icon="👋🏾")
            with st.expander(":rainbow[**Настройка вывода**]"):
                # Расширенные настройки (для любопытных умников!)
                width = st.number_input("Ширина изображения", value=1024)
                height = st.number_input("Высота изображения", value=1024)
                num_outputs = st.slider(
                    "Количество изображений", value=1, min_value=1, max_value=4)
                scheduler = st.selectbox('Планировщик', ('DDIM', 'DPMSolverMultistep', 'HeunDiscrete',
                                                       'KarrasDPM', 'K_EULER_ANCESTRAL', 'K_EULER', 'PNDM'))
                num_inference_steps = st.slider(
                    "Количество шагов денойзинга", value=50, min_value=1, max_value=500)
                guidance_scale = st.slider(
                    "Масштаб для классификатор-безопасного управления", value=7.5, min_value=1.0, max_value=50.0, step=0.1)
                prompt_strength = st.slider(
                    "Сила подсказки при использовании img2img/inpaint (1.0 соответствует полной утрате информации в изображении)", value=0.8, max_value=1.0, step=0.1)
                refine = st.selectbox(
                    "Выберите стиль уточнения (оставили только 2)", ("expert_ensemble_refiner", "None"))
                high_noise_frac = st.slider(
                    "Доля шума для `expert_ensemble_refiner`", value=0.8, max_value=1.0, step=0.1)
            prompt = st.text_area(
                ":orange[**Введите запрос: начните печатать**]",
                value="Астронавт, скачущий на радуге единорога, кинематографичный, драматичный")
            negative_prompt = st.text_area(":orange[**Что не должно быть на изображении?**]",
                                           value="абсолютно худшее качество, искаженные черты",
                                           help="Это негативная подсказка, укажите, что не хотите видеть на изображении")

            # Кнопка "Отправить"
            submitted = st.form_submit_button(
                "Отправить", type="primary", use_container_width=True)

        # Credits and resources
        st.divider()
        st.markdown(
            ":orange[**Ресурсы:**]  \n"
            f"<img src='{replicate_logo}' style='height: 1em'> [{replicate_text}]({replicate_link})",
            unsafe_allow_html=True
        )
        st.markdown(
            """
            ---
            Следите за мной в соцсетях:

            𝕏 → [@tonykipkemboi](https://twitter.com/tonykipkemboi)

            LinkedIn → [Tony Kipkemboi](https://www.linkedin.com/in/tonykipkemboi)

            """
        )

        return submitted, width, height, num_outputs, scheduler, num_inference_steps, guidance_scale, prompt_strength, refine, high_noise_frac, prompt, negative_prompt


def main_page(submitted: bool, width: int, height: int, num_outputs: int,
              scheduler: str, num_inference_steps: int, guidance_scale: float,
              prompt_strength: float, refine: str, high_noise_frac: float,
              prompt: str, negative_prompt: str) -> None:
    """Основной макет страницы и логика генерации изображений.

    Args:
        submitted (bool): Флаг, указывающий, была ли отправлена форма.
        width (int): Ширина выходного изображения.
        height (int): Высота выходного изображения.
        num_outputs (int): Количество изображений для вывода.
        scheduler (str): Тип планировщика для модели.
        num_inference_steps (int): Количество шагов денойзинга.
        guidance_scale (float): Масштаб для классификатор-безопасного управления.
        prompt_strength (float): Сила подсказки при использовании img2img/inpaint.
        refine (str): Стиль уточнения.
        high_noise_frac (float): Доля шума для `expert_ensemble_refiner`.
        prompt (str): Текстовый запрос для генерации изображения.
        negative_prompt (str): Текстовый запрос для элементов, которые нужно избежать на изображении.
    """
    if submitted:
        with st.status('👩🏾‍🍳 Превращаем ваши слова в искусство...', expanded=True) as status:
            st.write("⚙️ Модель запущена")
            st.write("🙆‍♀️ Разомнитесь пока ждете")
            try:
                # Вызов API только если нажата кнопка "Отправить"
                if submitted:
                    # Вызов API replicate для получения изображения
                    with generated_images_placeholder.container():
                        all_images = []  # Список для хранения всех сгенерированных изображений
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
                                'Ваше изображение сгенерировано!', icon='😍')
                            # Сохранение сгенерированного изображения в состояние сеанса
                            st.session_state.generated_image = output

                            # Отображение изображения
                            for image in st.session_state.generated_image:
                                with st.container():
                                    st.image(image, caption="Сгенерированное изображение 🎈",
                                             use_column_width=True)
                                    # Добавление изображения в список
                                    all_images.append(image)

                                    response = requests.get(image)
                        # Сохранение всех сгенерированных изображений в состояние сеанса
                        st.session_state.all_images = all_images

                        # Создание объекта BytesIO
                        zip_io = io.BytesIO()

                        # Опция загрузки для каждого изображения
                        with zipfile.ZipFile(zip_io, 'w') as zipf:
                            for i, image in enumerate(st.session_state.all_images):
                                response = requests.get(image)
                                if response.status_code == 200:
                                    image_data = response.content
                                    # Запись каждого изображения в zip-файл с именем
                                    zipf.writestr(
                                        f"output_file_{i+1}.png", image_data)
                                else:
                                    st.error(
                                        f"Не удалось получить изображение {i+1} с {image}. Код ошибки: {response.status_code}", icon="🚨")
                        # Создание кнопки загрузки для zip-файла
                        st.download_button(
                            ":red[**Скачать все изображения**]", data=zip_io.getvalue(), file_name="output_files.zip", mime="application/zip", use_container_width=True)
                status.update(label="✅ Изображения сгенерированы!",
                              state="complete", expanded=False)
            except Exception as e:
                print(e)
                st.error(f'Произошла ошибка: {e}', icon="🚨")

    # Если не отправлено, расслабьтесь 🍹
    else:
        pass

    # Отображение галереи для вдохновения
    with gallery_placeholder.container():
        img = image_select(
            label="Понравилось? Сохраните! 😉",
            images=[
                "gallery/farmer_sunset.png", "gallery/astro_on_unicorn.png",
                "gallery/friends.png", "gallery/wizard.png", "gallery/puppy.png",
                "gallery/cheetah.png", "gallery/viking.png",
            ],
            captions=["Фермер на закате, кинематографичный, драматичный",
                      "Астронавт, скачущий на радуге единорога, кинематографичный, драматичный",
                      "Группа друзей смеется и танцует на музыкальном фестивале, радостная атмосфера, 35мм пленочная фотография",
                      "Волшебник, творящий заклинание, интенсивная магическая энергия, исходящая из его рук, чрезвычайно детализированная фантазийная иллюстрация",
                      "Милый щенок, играющий в поле цветов, малая глубина резкости, фотография Canon",
                      "Мать-гепард кормит своих детенышей в высокой траве Серенгети. Раннее утреннее солнце пробивается через траву. Фотография National Geographic от Франса Лантина",
                      "Портрет викинга-бородача в шлеме с рогами. Он пристально смотрит вдаль, держа боевой топор. Драматическое освещение, цифровая масляная живопись",
                      ],
            use_container_width=True
        )


def main():
    """
    Основная функция для запуска приложения Streamlit.

    Эта функция инициализирует настройку боковой панели и макет главной страницы.
    Она получает пользовательские вводы с боковой панели и передает их в основную функцию страницы.
    Основная функция страницы затем генерирует изображения на основе этих вводов.
    """
    submitted, width, height, num_outputs, scheduler, num_inference_steps, guidance_scale, prompt_strength, refine, high_noise_frac, prompt, negative_prompt = configure_sidebar()
    main_page(submitted, width, height, num_outputs, scheduler, num_inference_steps,
              guidance_scale, prompt_strength, refine, high_noise_frac, prompt, negative_prompt)


if __name__ == "__main__":
    main()
