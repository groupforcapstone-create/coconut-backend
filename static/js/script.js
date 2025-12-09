// ================= FIREBASE SETUP =================
const auth = firebase.auth();
const db = firebase.firestore();

// ===================== CONSTANTS =====================
const ADMIN_EMAIL = "groupforcapstone@gmail.com";
const ADMIN_NAME = "Jamesy Acolicol";
const ADMIN_ROLE = "Admin";
const ADMIN_ADDRESS = "MacArthur, Leyte";
const ADMIN_CONTACT = "09289230563";
const ADMIN_FARM = "My Farm";

// ===================== REGISTER =====================
async function register() {
  const name = document.getElementById('register-name').value.trim();
  const email = document.getElementById('register-email').value.trim();
  const password = document.getElementById('register-password').value.trim();
  const address = document.getElementById('register-address').value.trim();
  const contact = document.getElementById('register-contact').value.trim();
  const farmName = document.getElementById('register-farm').value.trim();
  const errorDiv = document.getElementById('register-error');
  errorDiv.textContent = "";

  if (!name || !email || !password) {
    errorDiv.textContent = "⚠️ Name, Email, and Password are required!";
    return;
  }

  try {
    const userCredential = await auth.createUserWithEmailAndPassword(email, password);
    const userId = userCredential.user.uid;

    // If registering admin account, force admin details
    const isAdmin = email === ADMIN_EMAIL;
    const sellerData = {
      name: isAdmin ? ADMIN_NAME : name,
      email: email,
      address: isAdmin ? ADMIN_ADDRESS : address,
      contact: isAdmin ? ADMIN_CONTACT : contact,
      farmName: isAdmin ? ADMIN_FARM : farmName,
      role: isAdmin ? ADMIN_ROLE : "User"
    };

    await db.collection("sellers").doc(userId).set(sellerData);

    alert("✅ Registered successfully! Please login.");
    window.location.href = "index.html";
  } catch (err) {
    errorDiv.textContent = "⚠️ " + err.message;
  }
}

// ===================== LOGIN =====================
async function login() {
  const email = document.getElementById('login-email').value.trim();
  const password = document.getElementById('login-password').value.trim();
  const errorDiv = document.getElementById('login-error');
  errorDiv.textContent = "";

  if (!email || !password) {
    errorDiv.textContent = "⚠️ Email and Password are required!";
    return;
  }

  try {
    const userCredential = await auth.signInWithEmailAndPassword(email, password);
    const userId = userCredential.user.uid;

    const doc = await db.collection("sellers").doc(userId).get();
    if (!doc.exists) {
      errorDiv.textContent = "⚠️ Seller info not found!";
      return;
    }

    const sellerData = doc.data();
    localStorage.setItem("sellerId", userId);
    localStorage.setItem("sellerData", JSON.stringify(sellerData));

    window.location.href = sellerData.role === "Admin" ? "admin.html" : "dashboard.html";
  } catch (err) {
    errorDiv.textContent = "⚠️ " + err.message;
  }
}

// ===================== DASHBOARD =====================
let sellerData = JSON.parse(localStorage.getItem("sellerData")) || {};

function updateSellerDisplay() {
  document.getElementById("display-name").textContent = sellerData.name || "Seller";
  document.getElementById("display-address").textContent = sellerData.address || "N/A";
  document.getElementById("display-contact").textContent = sellerData.contact || "N/A";
  document.getElementById("display-farm").textContent = sellerData.farmName || "N/A";
  document.getElementById("display-role").textContent = sellerData.role || "User";

  document.getElementById("seller-name-input").value = sellerData.name || "";
  document.getElementById("seller-address-input").value = sellerData.address || "";
  document.getElementById("seller-contact-input").value = sellerData.contact || "";
  document.getElementById("seller-farm-input").value = sellerData.farmName || "";
}

// ===================== UPDATE SELLER INFO =====================
async function updateSellerInfo() {
  const name = document.getElementById("seller-name-input").value.trim();
  const address = document.getElementById("seller-address-input").value.trim();
  const contact = document.getElementById("seller-contact-input").value.trim();
  const farmName = document.getElementById("seller-farm-input").value.trim();
  const userId = localStorage.getItem("sellerId");

  sellerData = { ...sellerData, name, address, contact, farmName };

  try {
    await db.collection("sellers").doc(userId).update(sellerData);
    localStorage.setItem("sellerData", JSON.stringify(sellerData));
    updateSellerDisplay();
    alert("✅ Seller info updated!");
  } catch (err) {
    alert("⚠️ " + err.message);
  }
}

// ===================== PRODUCT MANAGEMENT =====================
async function addProduct() {
  const name = document.getElementById("product-name").value.trim();
  const price = document.getElementById("product-price").value.trim();
  const quantity = document.getElementById("product-quantity").value.trim();
  const description = document.getElementById("product-description").value.trim();
  const imageInput = document.getElementById("product-image");
  const userId = localStorage.getItem("sellerId");
  let imageUrl = "";

  if (!name || !price || !quantity) {
    alert("⚠️ Name, price, and quantity are required!");
    return;
  }

  if (imageInput.files.length > 0) {
    const file = imageInput.files[0];
    const reader = new FileReader();
    reader.onload = async (e) => {
      imageUrl = e.target.result;
      await saveProduct();
    };
    reader.readAsDataURL(file);
  } else {
    await saveProduct();
  }

  async function saveProduct() {
    const product = { name, price, quantity, description, imageUrl };
    try {
      await db.collection("sellers").doc(userId).collection("products").doc().set(product);
      alert("✅ Product added!");
      renderProducts();
    } catch (err) {
      alert("⚠️ " + err.message);
    }
  }
}

async function renderProducts() {
  const tbody = document.querySelector("#product-table tbody");
  tbody.innerHTML = "";
  const userId = localStorage.getItem("sellerId");

  try {
    const snapshot = await db.collection("sellers").doc(userId).collection("products").get();
    snapshot.forEach(doc => {
      const p = doc.data();
      const row = document.createElement("tr");
      row.innerHTML = `
        <td>${p.imageUrl ? `<img src="${p.imageUrl}" class="product-img">` : "No Image"}</td>
        <td>${p.name}</td>
        <td>${p.price}</td>
        <td>${p.quantity}</td>
        <td>${p.description}</td>
        <td>
          <button onclick="editProduct('${doc.id}')">Edit</button>
          <button onclick="deleteProduct('${doc.id}')">Delete</button>
        </td>`;
      tbody.appendChild(row);
    });
  } catch (err) {
    alert("⚠️ " + err.message);
  }
}

// ===================== LOGOUT & DELETE =====================
function logout() {
  auth.signOut().then(() => {
    localStorage.clear();
    window.location.href = "index.html";
  });
}

async function deleteAccount() {
  if (!confirm("⚠️ Delete account? This will remove all data.")) return;
  const userId = localStorage.getItem("sellerId");

  try {
    const productsSnap = await db.collection("sellers").doc(userId).collection("products").get();
    for (const doc of productsSnap.docs) await doc.ref.delete();
    await db.collection("sellers").doc(userId).delete();
    await auth.currentUser.delete();

    localStorage.clear();
    window.location.href = "index.html";
  } catch (err) {
    alert("⚠️ " + err.message);
  }
}

// ===================== INITIALIZE =====================
document.addEventListener("DOMContentLoaded", () => {
  renderProducts();
  updateSellerDisplay();
});
