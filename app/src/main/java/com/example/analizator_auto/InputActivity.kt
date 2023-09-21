package com.example.analizator_auto

import android.content.Intent
import android.content.pm.PackageManager
import android.net.Uri
import androidx.appcompat.app.AppCompatActivity
import android.os.Bundle
import android.widget.Toast
import androidx.core.content.ContextCompat
import com.example.analizator_auto.databinding.ActivityInputBinding


class InputActivity : AppCompatActivity() {
    private lateinit var binding: ActivityInputBinding

    private var uri: Uri? = null


    override fun onCreate(savedInstanceState: Bundle?) {
        binding = ActivityInputBinding.inflate(layoutInflater)
        super.onCreate(savedInstanceState)
        setContentView(binding.root)

        binding.userImage.setImageResource(android.R.drawable.ic_menu_add)

        // Кнопка "Добавить изображение" (id: "btInputImage") - получение изображения и его вывод на экран
        binding.btInputImage.setOnClickListener {

            // Если разрешения еще нет
            if (ContextCompat.checkSelfPermission(this, android.Manifest.permission.READ_EXTERNAL_STORAGE)
                == PackageManager.PERMISSION_DENIED) {

                val permissions = arrayOf(android.Manifest.permission.READ_EXTERNAL_STORAGE)
                requestPermissions(permissions, Constance.PERMISSION_CODE)
            }

            // Если разрешение уже есть, то выполнить основную функцию
            else {
                pickImageFromGallery()
            }

        }


        // Аналогичные действия при нажатие на изображение (id: "userImage") - получение изображения и его вывод на экран
        binding.userImage.setOnClickListener {

            // Если разрешения еще нет
            if (ContextCompat.checkSelfPermission(this, android.Manifest.permission.READ_EXTERNAL_STORAGE)
                == PackageManager.PERMISSION_DENIED) {

                val permissions = arrayOf(android.Manifest.permission.READ_EXTERNAL_STORAGE)
                requestPermissions(permissions, Constance.PERMISSION_CODE)
            }

            // Если разрешение уже есть, то выполнить основную функцию
            else {
                pickImageFromGallery()
            }

        }


        // Кнопка "Готово" (id: "done") - переход на Result Activity с передачей изображения с проверкой
        binding.done.setOnClickListener {

            // Если пользователь не добавил изображение
            if (uri == null) {
                Toast.makeText(this, "Добавьте изображение", Toast.LENGTH_LONG).show()
            }

            // Если пользователь добавил изображение
            else {
                val i = Intent(this, ResultActivity::class.java)
                i.putExtra(Constance.IMAGE, uri)
                startActivity(i)
            }

        }

    }


    // Функция для выбора изображения
    private fun pickImageFromGallery() {
        val intent = Intent()
        intent.setAction(Intent.ACTION_GET_CONTENT)
        intent.setType("image/*")

        startActivityForResult(intent, Constance.PERMISSION_CODE)

    }


    @Deprecated("Deprecated in Java")
    // Функция для визуализации полученного изображения
    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        super.onActivityResult(requestCode, resultCode, data)

        if (resultCode == RESULT_OK && null != data){
            uri = data.data!!

            if (requestCode == Constance.PERMISSION_CODE) {
                binding.userImage.setImageURI(uri)
            }
        }

    }
}