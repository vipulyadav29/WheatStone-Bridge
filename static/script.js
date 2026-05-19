const fileInput = document.getElementById("leafImage");
const uploadBox = document.querySelector(".upload-box");

if (fileInput && uploadBox) {
  fileInput.addEventListener("change", () => {
    const file = fileInput.files && fileInput.files[0];
    if (!file) {
      return;
    }

    const info = uploadBox.querySelector("small");
    if (info) {
      info.textContent = `Selected: ${file.name}`;
    }
  });
}
