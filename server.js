import express from "express";
import cors from "cors";
import dotenv from "dotenv";
import bodyParser from "body-parser";
import path from "path";
import mysql from "mysql2/promise";
import jwt from "jsonwebtoken";
import { fileURLToPath } from "url";

// Initialize environment variables
dotenv.config();

// Set __dirname for ES module compatibility
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// Initialize Express app
const app = express();

// === MYSQL DATABASE SETUP ===
const db = mysql.createPool({
  host: process.env.DB_HOST,
  user: process.env.DB_USER,
  password: process.env.DB_PASSWORD,
  database: process.env.DB_NAME,
  waitForConnections: true,
  connectionLimit: 10,
  queueLimit: 0,
});

// === TEST DATABASE CONNECTION ===
(async () => {
  try {
    const conn = await db.getConnection();
    console.log("✅ Connected to MySQL!");
    conn.release();
  } catch (err) {
    console.error("❌ Database Connection Error:", err.message);
    process.exit(1);
  }
})();

// === EXPRESS MIDDLEWARE ===
app.use(cors({
  origin: ["http://localhost:5502", "http://127.0.0.1:5502"],
  methods: ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
  allowedHeaders: ["Content-Type", "Authorization"],
  credentials: true,
  preflightContinue: false, // Add this
  optionsSuccessStatus: 204 // Add this
}));

app.use(bodyParser.json());
app.use(express.static(path.join(__dirname, "public")));

// === AUTH MIDDLEWARE ===
const verifyToken = (roles = []) => {
  return (req, res, next) => {
    const authHeader = req.headers.authorization;
    if (!authHeader) return res.status(401).json({ error: "Missing Authorization header" });

    const token = authHeader.split(" ")[1];
    if (!token) return res.status(401).json({ error: "Authorization token is missing" });

    jwt.verify(token, process.env.JWT_SECRET, (err, decoded) => {
      if (err) return res.status(403).json({ error: "Invalid or expired token" });

      if (roles.length && !roles.includes(decoded.role)) {
        return res.status(403).json({ error: "Access denied: Role not allowed" });
      }

      req.userId = decoded.id;
      req.userRole = decoded.role;
      next();
    });
  };
};

// === SIGNUP ROUTE ===
app.post('/api/auth/signup', async (req, res) => {
  const { username, password, email, role } = req.body;

  if (!username || !password || !email || !role) {
    return res.status(400).json({ error: "All fields are required" });
  }

  try {
    const [existing] = await db.query("SELECT * FROM users WHERE username = ?", [username]);
    if (existing.length > 0) {
      return res.status(409).json({ error: "Username already exists" });
    }

    await db.query(
      "INSERT INTO users (username, password, email, role) VALUES (?, ?, ?, ?)",
      [username, password, email, role]
    );

    const [userRow] = await db.query("SELECT * FROM users WHERE username = ?", [username]);
    const user = userRow[0];
    const token = jwt.sign({ id: user.id, role: user.role }, process.env.JWT_SECRET, { expiresIn: "1h" });

    res.status(201).json({ message: "Signup successful", token });
  } catch (error) {
    console.error("❌ Signup error:", error);
    res.status(500).json({ error: "Internal server error" });
  }
});

// === LOGIN ROUTE ===
app.post('/api/auth/login', async (req, res) => {
  const { email, password } = req.body;

  try {
    const [rows] = await db.query("SELECT * FROM users WHERE email = ?", [email]);
    const user = rows[0];

    if (!user) return res.status(404).json({ error: "User not found" });

    const validPassword = password === user.password; // Replace with bcrypt.compare for hashed passwords
    if (!validPassword) return res.status(401).json({ error: "Invalid password" });

    const token = jwt.sign({ id: user.id, role: user.role }, process.env.JWT_SECRET, { expiresIn: "1h" });

    res.json({
      token,
      username: user.username, // Add username here
      role: user.role, // Add role here
    });
  } catch (error) {
    console.error("❌ Login error:", error);
    res.status(500).json({ error: "Internal server error" });
  }
});
// === DELETE FOOD ENTRY ===
app.delete("/admin/food-entries/:id", verifyToken(["admin"]), async (req, res) => {
  const { id } = req.params; // Extract the food item ID from the URL parameters

  try {
    // Attempt to delete the food entry from the database
    const [result] = await db.query("DELETE FROM food_entries WHERE id = ?", [id]);

    if (result.affectedRows === 0) {
      // If no rows were affected, it means no food entry with that ID was found
      console.log(`❌ Food entry with ID ${id} not found for deletion.`);
      return res.status(404).json({ message: "Food entry not found." });
    }

    // If deletion was successful
    console.log(`✅ Food entry with ID ${id} deleted successfully.`);
    res.status(200).json({ message: "Food entry deleted successfully." });
  } catch (err) {
    // Catch any database or unexpected errors
    console.error("❌ Delete food entry error:", err);
    res.status(500).json({ message: "Failed to delete food entry due to an internal server error." });
  }
});
// === ADD FOOD ROUTE ===
app.post("/admin/add-food", verifyToken(["admin"]), async (req, res) => {
  const { food_name, carbs, protein, fat } = req.body;

  if (!food_name || isNaN(carbs) || isNaN(protein) || isNaN(fat)) {
    return res.status(400).json({ message: "Invalid food data" });
  }

  const calories = (carbs * 4) + (protein * 4) + (fat * 9);

  try {
    const [result] = await db.query(
      "INSERT INTO food_entries (food_name, carbs, protein, fat, calories) VALUES (?, ?, ?, ?, ?)",
      [food_name, carbs, protein, fat, calories]
    );

    res.status(200).json({ message: "Food added successfully", id: result.insertId });
  } catch (err) {
    console.error("❌ Database insert error:", err);
    res.status(500).json({ message: "Database error" });
  }
});

// === Get All Food Entries ===
app.get("/admin/food-entries", verifyToken(["admin"]), async (req, res) => {
  try {
    const [rows] = await db.query("SELECT id, food_name, carbs, protein, fat, calories FROM food_entries");
    res.json(rows);
  } catch (error) {
    console.error("❌ Fetch food error:", error);
    res.status(500).json({ error: "Error fetching food entries" });
  }
});

app.get("/admin/admins", verifyToken(["admin"]), async (req, res) => {
  try {
    const [admins] = await db.query("SELECT id, username, email FROM users WHERE role = 'admin'");
    res.json(admins);
  } catch (err) {
    console.error("❌ Error fetching admins:", err);
    res.status(500).json({ error: "Failed to fetch admins" });
  }
});

app.get("/admin/users", verifyToken(["admin"]), async (req, res) => {
  try {
    const [users] = await db.query("SELECT id, username, email FROM users WHERE role = 'user'");
    res.json(users);
  } catch (err) {
    console.error("❌ Error fetching users:", err);
    res.status(500).json({ error: "Failed to fetch users" });
  }
});
// === DELETE FOOD ENTRY ===

// === DELETE ADMIN ===

app.post("/admin/add-admin", verifyToken(["admin"]), async (req, res) => {
  const { username, email, password } = req.body;

  if (!username || !email || !password) {
    return res.status(400).json({ message: "All fields are required" });
  }

  try {
    // Check if admin already exists
    const [existing] = await db.query("SELECT id FROM users WHERE email = ?", [email]);
    if (existing.length > 0) {
      return res.status(400).json({ message: "Admin already exists with this email" });
    }

    // You should hash the password in production
    const [result] = await db.query(
      "INSERT INTO users (username, email, password, role) VALUES (?, ?, ?, 'admin')",
      [username, email, password]
    );

    res.status(201).json({ message: "Admin added successfully", id: result.insertId });
  } catch (err) {
    console.error("❌ Error adding admin:", err);
    res.status(500).json({ message: "Failed to add admin" });
  }
});

app.delete("/admin/admins/:id", verifyToken(["admin"]), async (req, res) => {
    const { id } = req.params;
    const adminIdToDelete = parseInt(id);
    const currentUserId = parseInt(req.userId); // req.userId is set by verifyToken middleware

    console.log("🔍 Delete Admin Debug:");
    console.log("Current user ID:", currentUserId, "Type:", typeof currentUserId);
    console.log("Admin ID to delete:", adminIdToDelete, "Type:", typeof adminIdToDelete);

    try {
        // Prevent deleting the currently logged-in admin
        if (currentUserId === adminIdToDelete) {
            console.log("❌ Self-deletion attempt blocked");
            return res.status(400).json({ message: "You cannot delete your own account" });
        }

        // First check if the admin exists and has the 'admin' role
        const [checkResult] = await db.query("SELECT id FROM users WHERE id = ? AND role = 'admin'", [adminIdToDelete]);
        if (checkResult.length === 0) {
            console.log("❌ Admin not found or is not an admin");
            return res.status(404).json({ message: "Admin not found or you cannot delete this user (not an admin)" });
        }

        // Perform the deletion
        const [result] = await db.query("DELETE FROM users WHERE id = ? AND role = 'admin'", [adminIdToDelete]);
        console.log("✅ SQL Delete Result:", result);

        if (result.affectedRows === 0) {
            // This case should ideally be caught by the checkResult above,
            // but as a fallback for race conditions or unexpected issues.
            console.log("❌ No rows affected during deletion (admin might have been removed concurrently)");
            return res.status(404).json({ message: "Failed to delete admin: Admin not found or already deleted" });
        }

        console.log("✅ Admin deleted successfully");
        res.status(200).json({ message: "Admin deleted successfully" });
    } catch (err) {
        console.error("❌ Delete admin error:", err);
        // Generic 500 for database or unexpected errors
        res.status(500).json({ message: "Failed to delete admin due to an internal server error." });
    }
});

// === SERVER SETUP ===
const PORT = process.env.PORT || 3000;
app.listen(PORT, () => console.log(`🚀 Server running at http://127.0.0.1:${PORT}`));

// Cleanup on server shutdown
process.on("SIGINT", async () => {
  console.log("\n🛑 Closing MySQL connection...");
  await db.end();
  console.log("✅ MySQL connection closed. Exiting.");
  process.exit(0);
});
