/**
 * Adds a new task to the task list.
 * @param {Array<Object>} taskList - The current array of tasks.
 * @param {Object} newTask - The new task object to add.
 * @returns {Array<Object>} The updated task list.
 */
export function addTaskToTaskList(taskList, newTask) {
  if (typeof taskList !== 'object' || taskList === null || !Array.isArray(taskList)) {
    console.error("Invalid taskList provided. Expected an array.");
    return [];
  }
  if (typeof newTask !== 'object' || newTask === null) {
    console.error("Invalid newTask provided. Expected an object.");
    return taskList;
  }
  
  // Check for duplicates (optional, but good practice)
  if (taskList.some(task => task.id === newTask.id)) {
    console.warn(`Task with ID ${newTask.id} already exists and was not added.`);
    return taskList;
  }

  const updatedList = [...taskList, newTask];
  return updatedList;
}