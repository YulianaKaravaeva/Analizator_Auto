package com.example.analizator_auto

import android.content.Intent
import androidx.appcompat.app.AppCompatActivity
import android.os.Bundle
import com.example.analizator_auto.databinding.ActivityMainBinding

class MainActivity : AppCompatActivity() {
    private lateinit var binding: ActivityMainBinding

    override fun onCreate(savedInstanceState: Bundle?) {
        binding = ActivityMainBinding.inflate(layoutInflater)
        super.onCreate(savedInstanceState)
        setContentView(binding.root)

        // Кнопка "Начать" (id: "start") - переход на Input Activity без передачи данных
        binding.start.setOnClickListener {
            val i = Intent(this, InputActivity::class.java)
            startActivity(i)
        }
    }
}