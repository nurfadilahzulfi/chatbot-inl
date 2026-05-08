import qrcode
from PIL import Image


def generate_qr_with_logo(
    data: str,
    logo_path: str,
    filename: str = "qrcode_logo.png"
):
    """
    Generate QR Code dengan logo di tengah

    Parameters:
    ----------
    data : str
        URL / text
    logo_path : str
        Path ke file logo (PNG disarankan)
    filename : str
        Output file
    """

    # Step 1: Generate QR
    qr = qrcode.QRCode(
        version=None,
        error_correction=qrcode.constants.ERROR_CORRECT_H,  # penting!
        box_size=10,
        border=4
    )

    qr.add_data(data)
    qr.make(fit=True)

    qr_img = qr.make_image(fill_color="black", back_color="white").convert("RGB")

    # Step 2: Load logo
    logo = Image.open(logo_path)

    # Step 3: Resize logo (maks 20-25% dari QR)
    qr_width, qr_height = qr_img.size

    logo_size = int(qr_width * 0.25)
    logo = logo.resize((logo_size, logo_size), Image.LANCZOS)

    # Step 4: Posisi tengah
    pos = (
        (qr_width - logo_size) // 2,
        (qr_height - logo_size) // 2
    )

    # Step 5: Paste logo ke QR
    qr_img.paste(logo, pos, mask=logo if logo.mode == 'RGBA' else None)

    # Step 6: Save
    qr_img.save(filename)

    print(f"QR Code + logo berhasil dibuat: {filename}")


if __name__ == "__main__":
    url = "https://warung-tisya.vercel.app/"
    logo_path = "logo.jpeg"  # ganti sesuai file kamu

    generate_qr_with_logo(url, logo_path, "qr_warung_logo.png")