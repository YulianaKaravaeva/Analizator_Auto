package com.example.analizator_auto

import android.content.Intent
import androidx.appcompat.app.AppCompatActivity
import android.os.Bundle
import com.example.analizator_auto.databinding.ActivityResultBinding
import android.content.ContentValues
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Matrix
import android.graphics.Paint
import android.media.ExifInterface
import android.net.Uri
import android.os.Build
import android.os.Environment
import android.provider.MediaStore
import android.util.Log
import android.widget.Toast
import androidx.core.content.ContextCompat
import java.io.File
import java.io.FileOutputStream
import java.io.IOException
import java.io.OutputStream

class ResultActivity : AppCompatActivity() {
    private lateinit var binding: ActivityResultBinding

    private var uri: Uri? = null
    private var bitmap: Bitmap? = null
    private var rgba: Bitmap? = null
    private val yolov8: Yolov8 = Yolov8()

    override fun onCreate(savedInstanceState: Bundle?) {
        binding = ActivityResultBinding.inflate(layoutInflater)
        super.onCreate(savedInstanceState)
        setContentView(binding.root)


        // Получение данных (изображения) с Input Activity
        uri = intent.getParcelableExtra<Uri?>(Constance.IMAGE)


        // Работа с нейросетью

        // Инициализация нейросети
        val ret_init: Boolean = yolov8.Init(assets)
        if (!ret_init) {
            Log.e("MainActivity", "yolov8 Init failed")
        }

        // Преобразование в bitmap
        bitmap = uri?.let { decodeUri(it) }
        val yourSelectedImage = bitmap!!.copy(Bitmap.Config.ARGB_8888, true)
        binding.resultImage.setImageBitmap(bitmap)

        // Вызов функции обработки изображения
        val objects: Array<Yolov8.Obj?>? = yolov8.Detect(yourSelectedImage, false)
        showObjects(objects)


        // Кнопка "Сохранить" (id: "save") - сохранение данных (изображения) в памяь телефона
        binding.save.setOnClickListener {

            // Если разрешения еще нет
            if (ContextCompat.checkSelfPermission(this, android.Manifest.permission.READ_EXTERNAL_STORAGE)
                == PackageManager.PERMISSION_DENIED) {

                val permissions = arrayOf(android.Manifest.permission.READ_EXTERNAL_STORAGE)
                requestPermissions(permissions, Constance.PERMISSION_CODE)
            }

            // Если разрешение уже есть, то выполнить основную функцию
            else {
                saveMediaToStorage()
            }

        }


        // Кнопка "OK" (id: "finish") - переход на Main Activity без передачи данных
        binding.finish.setOnClickListener {
            val i = Intent(this, MainActivity::class.java)
            startActivity(i)
        }

    }


    // Функция рисования рамок
    private fun showObjects(objects: Array<Yolov8.Obj?>?) {
        if (objects == null) {
            binding.resultImage.setImageBitmap(bitmap)
            return
        }

        // draw objects on bitmap
        val rgba = bitmap!!.copy(Bitmap.Config.ARGB_8888, true)
        val colors = intArrayOf(
            Color.rgb(54, 67, 244),
            Color.rgb(99, 30, 233),
            Color.rgb(176, 39, 156),
            Color.rgb(183, 58, 103),
            Color.rgb(181, 81, 63),
            Color.rgb(243, 150, 33),
            Color.rgb(244, 169, 3),
            Color.rgb(212, 188, 0),
            Color.rgb(136, 150, 0),
            Color.rgb(80, 175, 76),
            Color.rgb(74, 195, 139),
            Color.rgb(57, 220, 205),
            Color.rgb(59, 235, 255),
            Color.rgb(7, 193, 255),
            Color.rgb(0, 152, 255),
            Color.rgb(34, 87, 255),
            Color.rgb(72, 85, 121),
            Color.rgb(158, 158, 158),
            Color.rgb(139, 125, 96)
        )
        val canvas = Canvas(rgba)
        val paint = Paint()
        paint.style = Paint.Style.STROKE
        paint.strokeWidth = 4f
        val textbgpaint = Paint()
        textbgpaint.color = Color.WHITE
        textbgpaint.style = Paint.Style.FILL
        val textpaint = Paint()
        textpaint.color = Color.BLACK
        textpaint.textSize = 26f
        textpaint.textAlign = Paint.Align.LEFT
        for (i in objects.indices) {
            paint.color = colors[i % 19]
            canvas.drawRect(
                objects[i]!!.x,
                objects[i]!!.y,
                objects[i]!!.x + objects[i]!!.w,
                objects[i]!!.y + objects[i]!!.h,
                paint
            )

            // draw filled text inside image
            run {
                val text: String = objects[i]!!.label + " = " + String.format(
                    "%.1f",
                    objects[i]!!.prob * 100
                ) + "%"
                val text_width = textpaint.measureText(text)
                val text_height = -textpaint.ascent() + textpaint.descent()
                var x: Float = objects[i]!!.x
                var y: Float = objects[i]!!.y - text_height
                if (y < 0) y = 0f
                if (x + text_width > rgba.width) x = rgba.width - text_width
                canvas.drawRect(x, y, x + text_width, y + text_height, textbgpaint)
                canvas.drawText(text, x, y - textpaint.ascent(), textpaint)
            }
        }
        binding.resultImage.setImageBitmap(rgba)
    }

    // Раскадировка uri
    private fun decodeUri(selectedImage: Uri): Bitmap {
        // Decode image size
        val o = BitmapFactory.Options()
        o.inJustDecodeBounds = true
        BitmapFactory.decodeStream(contentResolver.openInputStream(selectedImage), null, o)

        // The new size we want to scale to
        val REQUIRED_SIZE = 640

        // Find the correct scale value. It should be the power of 2.
        var width_tmp = o.outWidth
        var height_tmp = o.outHeight
        var scale = 1
        while (true) {
            if (width_tmp / 2 < REQUIRED_SIZE
                || height_tmp / 2 < REQUIRED_SIZE
            ) {
                break
            }
            width_tmp /= 2
            height_tmp /= 2
            scale *= 2
        }

        // Decode with inSampleSize
        val o2 = BitmapFactory.Options()
        o2.inSampleSize = scale
        val bitmap =
            BitmapFactory.decodeStream(contentResolver.openInputStream(selectedImage), null, o2)

        // Rotate according to EXIF
        var rotate = 0
        try {
            val exif = ExifInterface(contentResolver.openInputStream(selectedImage)!!)
            val orientation = exif.getAttributeInt(
                ExifInterface.TAG_ORIENTATION,
                ExifInterface.ORIENTATION_NORMAL
            )
            when (orientation) {
                ExifInterface.ORIENTATION_ROTATE_270 -> rotate = 270
                ExifInterface.ORIENTATION_ROTATE_180 -> rotate = 180
                ExifInterface.ORIENTATION_ROTATE_90 -> rotate = 90
            }
        } catch (e: IOException) {
            Log.e("MainActivity", "ExifInterface IOException")
        }
        val matrix = Matrix()
        matrix.postRotate(rotate.toFloat())
        return Bitmap.createBitmap(bitmap!!, 0, 0, bitmap.width, bitmap.height, matrix, true)
    }



    // Функция сохранения данных
    fun saveMediaToStorage() {
        //Generating a file name
        val filename = "${System.currentTimeMillis()}.png"

        //Output stream
        var fos: OutputStream? = null

        //For devices running android >= Q
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
            //getting the contentResolver
            applicationContext.contentResolver.also { resolver ->

                //Content resolver will process the contentvalues
                val contentValues = ContentValues().apply {

                    //putting file information in content values
                    put(MediaStore.MediaColumns.DISPLAY_NAME, filename)
                    put(MediaStore.MediaColumns.MIME_TYPE, "image/png")
                    put(MediaStore.MediaColumns.RELATIVE_PATH, Environment.DIRECTORY_PICTURES)
                }

                //Inserting the contentValues to contentResolver and getting the Uri
                val imageUri: Uri? =
                    resolver.insert(MediaStore.Images.Media.EXTERNAL_CONTENT_URI, contentValues)

                //Opening an outputstream with the Uri that we got
                fos = imageUri?.let { resolver.openOutputStream(it) }
            }
        } else {
            //These for devices running on android < Q
            //So I don't think an explanation is needed here
            val imagesDir =
                Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_PICTURES)
            val image = File(imagesDir, filename)
            fos = FileOutputStream(image)
        }

        fos?.use {
            //Finally writing the bitmap to the output stream that we opened
            rgba?.compress(Bitmap.CompressFormat.PNG, 100, it)
            Toast.makeText(this, "Фото сохранено", Toast.LENGTH_LONG).show()
        }
    }

}