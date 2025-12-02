import napari


def copy_custom_ui_icons():

    import os
    import glob
    import shutil

    my_icon_dir = os.path.abspath("icons")
    napari_source_dir = os.path.dirname(napari.__file__)
    icons_source_path = os.path.join(napari_source_dir, 'resources', 'icons')

    my_icons = glob.glob(os.path.join(my_icon_dir,"*.svg"))
    for icon in my_icons:
        print(f"copying {icon} to {os.path.join(icons_source_path,os.path.split(icon)[-1])}")
        shutil.copy(icon, os.path.join(icons_source_path,os.path.split(icon)[-1]))


def customize_stylesheet(app):
    from napari.utils.theme import get_theme
    from napari._qt.qt_resources import get_stylesheet

    theme_id = "dark"
    theme = get_theme(theme_id)
    stylesheet = get_stylesheet(theme.id)

    # Append your custom CSS
    # Note: We use the prefix 'theme_dark:' which now contains your copied file
    my_buttons = ("\n\nQtViewerPushButton[mode=\"coordinate_axes\"] { "
                  "\n    image: url(\"theme_dark:/coordinate_axes.svg\");"
                  "\n}\n")
    stylesheet += my_buttons

    app.setStyleSheet(stylesheet)



def customize_ui_old(app):

    from napari.utils.theme import get_theme
    from napari._qt.qt_resources import get_stylesheet
    from qtpy.QtCore import QDir
    import os
    import shutil

    theme_id = "dark"
    theme = get_theme(theme_id)

    resource_prefix = f'theme_{theme_id}'
    search_paths = QDir.searchPaths(resource_prefix)
    print(f'ICON FOLDER: {search_paths}')

    my_icons = []# ["coordinate_axes.svg"]
    icon_dir = os.path.abspath("icons")

    if search_paths:
        # search_paths[0] is the absolute path to the temp folder (e.g., /tmp/napari_icons_xyz/)
        target_dir = search_paths[0]

        for icon in my_icons:
            # 4. Copy your custom SVG to that folder
            # Assuming your source file is in the current directory

            source_path = os.path.join(icon_dir, icon)
            dest_path = os.path.join(target_dir, icon)

            try:
                shutil.copy(source_path, dest_path)
                print(f"Injected {icon} into {resource_prefix}")
            except FileNotFoundError:
                print(f"Error: Could not find source icon at {source_path}")



        # 5. Build and Apply the Stylesheet
    stylesheet = get_stylesheet(theme.id)

    # Append your custom CSS
    # Note: We use the prefix 'theme_dark:' which now contains your copied file
    my_buttons = ("\n\nQtViewerPushButton[mode=\"coordinate_axes\"] { "
                  "\n    image: url(\"theme_dark:/coordinate_axes.svg\");"
                  "\n}\n")
    stylesheet += my_buttons

    app.setStyleSheet(stylesheet)

